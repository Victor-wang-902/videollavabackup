import os
import sys
import signal
import fsspec
import fire

from omegaconf import OmegaConf
from typing import List, Optional, Any
import numpy as np  # pylint: disable=unused-import

# Import necessary components from video2dataset
from video2dataset.logger import LoggerProcess
from video2dataset.data_writer import (
    WebDatasetSampleWriter,
    FilesSampleWriter,
    ParquetSampleWriter,
    TFRecordSampleWriter,
    DummySampleWriter,
)
from video2dataset.input_sharder import InputSharder
from video2dataset.output_sharder import OutputSharder
from video2dataset.distributor import (
    no_distributor,
    multiprocessing_distributor,
    pyspark_distributor,
    SlurmDistributor,
    SlurmShardSampler,
)
from video2dataset.workers import DownloadWorker, SubsetWorker, OpticalFlowWorker, CaptionWorker, WhisperWorker
from video2dataset.configs import CONFIGS
from video2dataset.types import EncodeFormats


def identity(x):
    return x


# Define your custom sample writer
import json
import numpy as np


class CustomSampleWriter(FilesSampleWriter):
    def write(self, streams, key, caption, meta):
        subfolder = os.path.dirname(self.subfolder)
        # Extract 'videoid' and 'page_dir' from meta
        videoid = meta.get('videoid')
        page_dir = meta.get('page_dir')
        if videoid is None or page_dir is None:
            raise ValueError("Metadata must contain 'videoid' and 'page_dir'.")

        # Construct the output path using self.subfolder
        output_path = f"{subfolder}/{page_dir}"#/{videoid}.mp4"

        # Ensure the directory exists
        if not self.fs.exists(output_path):
            self.fs.makedirs(output_path)

        # Save the video file
        video_data = streams.get('video')
        if video_data:
            video_ext = self.encode_formats.get('video', 'mp4')
            video_filename = f"{output_path}/{videoid}.{video_ext}"
            with self.fs.open(video_filename, 'wb') as f:
                f.write(video_data)
        else:
            raise ValueError("No video data found in streams.")

        # Save the caption if present
        #if self.save_caption and caption:
        #    caption_filename = f"{output_path}/caption.txt"
        #    with self.fs.open(caption_filename, 'w') as f:
        #        f.write(str(caption))

        # Save additional metadata if necessary
        # Convert any non-JSON-serializable items in meta
        #for k, v in meta.items():
        #    if isinstance(v, np.ndarray):
        #        meta[k] = v.tolist()

        # Optionally save meta as JSON file
        #meta_filename = f"{output_path}/meta.json"
        #with self.fs.open(meta_filename, 'w') as f:
        #    json.dump(meta, f, indent=4)

        # Write metadata to Parquet file
        self.buffered_parquet_writer.write(meta)



# Modify the video2dataset function to accept a custom sample writer
def custom_video2dataset(
    url_list: str,
    output_folder: str = "dataset",
    output_format: str = "files",
    input_format: str = "csv",
    encode_formats: Optional[EncodeFormats] = None,
    stage: str = "download",
    url_col: str = "url",
    caption_col: Optional[str] = None,
    clip_col: Optional[str] = None,
    save_additional_columns: Optional[List[str]] = None,
    enable_wandb: bool = False,
    wandb_project: str = "video2dataset",
    incremental_mode: str = "incremental",
    max_shard_retry: int = 1e10,
    tmp_dir: str = "/tmp",
    config: Any = "default",
    custom_sample_writer_class=None,
):
    """
    Custom video2dataset function with support for a custom sample writer.
    """
    local_args = dict(locals())
    if isinstance(config, str):
        config = CONFIGS[config] if config in CONFIGS else OmegaConf.load(config)
        config = OmegaConf.to_container(config)
    for arg_type in ["subsampling", "reading", "storage", "distribution"]:
        assert arg_type in config

    if config["reading"]["sampler"] is None:
        config["reading"]["sampler"] = identity

    called_from_slurm = "CALLED_FROM_SLURM" in os.environ
    if called_from_slurm:
        global_task_id = int(os.environ["GLOBAL_RANK"])
        num_tasks = (
            config["distribution"]["distributor_args"]["n_nodes"]
            * config["distribution"]["distributor_args"]["tasks_per_node"]
        )
        config["reading"]["sampler"] = SlurmShardSampler(global_task_id=global_task_id, num_tasks=num_tasks)
        config["distribution"]["distributor"] = "multiprocessing"

        # Only log from master
        enable_wandb = enable_wandb and (global_task_id == 0)

    # Handle captions as subtitles if specified
    if config["storage"].get("captions_are_subtitles", False):
        assert clip_col is None  # no double-clipping
        if config["reading"]["yt_args"]["yt_metadata_args"] is None:
            config["reading"]["yt_args"]["yt_metadata_args"] = {}
        if not config["reading"]["yt_args"]["yt_metadata_args"].get("writesubtitles", None):  # type: ignore
            config["reading"]["yt_args"]["yt_metadata_args"]["writesubtitles"] = "all"  # type: ignore

    if encode_formats is None:
        encode_formats = {"video": "mp4"}

    def make_path_absolute(path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    output_folder = make_path_absolute(output_folder)
    url_list = make_path_absolute(url_list)

    logger_process = LoggerProcess(output_folder, enable_wandb, wandb_project, local_args)
    tmp_path = output_folder + "/_tmp"
    fs, run_tmp_dir = fsspec.core.url_to_fs(tmp_path)
    if not fs.exists(run_tmp_dir):
        fs.mkdir(run_tmp_dir)

    def signal_handler(signal_arg, frame):  # pylint: disable=unused-argument
        try:
            fs.rm(run_tmp_dir, recursive=True)
        except Exception as _:  # pylint: disable=broad-except
            pass
        logger_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    save_caption = caption_col is not None or config["storage"].get("captions_are_subtitles", False)

    fs, output_path = fsspec.core.url_to_fs(output_folder)

    if not fs.exists(output_path):
        fs.mkdir(output_path)
        done_shards = set()
    else:
        if incremental_mode == "incremental":
            done_shards = set(int(x.split("/")[-1].split("_")[0]) for x in fs.glob(output_path + "/*.json"))
        elif incremental_mode == "overwrite":
            fs.rm(output_path, recursive=True)
            fs.mkdir(output_path)
            done_shards = set()
        else:
            raise ValueError(f"Unknown incremental mode {incremental_mode}")

    logger_process.done_shards = done_shards
    logger_process.start()

    if custom_sample_writer_class is not None:
        sample_writer_class = custom_sample_writer_class
    else:
        if output_format == "webdataset":
            sample_writer_class = WebDatasetSampleWriter
        elif output_format == "parquet":
            sample_writer_class = ParquetSampleWriter  # type: ignore
        elif output_format == "files":
            sample_writer_class = FilesSampleWriter  # type: ignore
        elif output_format == "tfrecord":
            sample_writer_class = TFRecordSampleWriter  # type: ignore
        elif output_format == "dummy":
            sample_writer_class = DummySampleWriter  # type: ignore
        else:
            raise ValueError(f"Invalid output format {output_format}")

    if input_format == "webdataset":
        shard_iterator = OutputSharder(  # type: ignore
            url_list, input_format, done_shards, sampler=config["reading"]["sampler"]
        )
    else:
        shard_iterator = InputSharder(  # type: ignore
            url_list,
            input_format,
            url_col,
            caption_col,
            clip_col,
            save_additional_columns,
            config["storage"]["number_sample_per_shard"],
            done_shards,
            tmp_path,
            config["reading"]["sampler"],
        )

    if stage == "download":
        worker = DownloadWorker(
            sample_writer_class=sample_writer_class,
            save_caption=save_caption,
            output_folder=output_folder,
            column_list=shard_iterator.column_list,
            tmp_dir=tmp_dir,
            encode_formats=encode_formats,
            config=config,
        )
    elif stage == "subset":
        worker = SubsetWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            encode_formats=encode_formats,
            config=config,
        )
    elif stage == "optical_flow":
        is_slurm_task = "GLOBAL_RANK" in os.environ and config["distribution"]["distributor"] == "multiprocessing"
        worker = OpticalFlowWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            encode_formats=encode_formats,
            is_slurm_task=is_slurm_task,
            config=config,
        )
    elif stage == "caption":
        is_slurm_task = "GLOBAL_RANK" in os.environ and config["distribution"]["distributor"] == "multiprocessing"
        worker = CaptionWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            encode_formats=encode_formats,
            is_slurm_task=is_slurm_task,
            config=config,
        )
    elif stage == "whisper":
        is_slurm_task = "GLOBAL_RANK" in os.environ and config["distribution"]["distributor"] == "multiprocessing"
        worker = WhisperWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            column_list=shard_iterator.column_list,
            tmp_dir=tmp_dir,
            encode_formats=encode_formats,
            is_slurm_task=is_slurm_task,
            config=config,
        )
    else:
        raise ValueError(f"Invalid stage: {stage}")

    print("Starting the downloading of this file")
    if config["distribution"]["distributor"] == "multiprocessing" or called_from_slurm:
        distributor_fn = multiprocessing_distributor if stage not in ["whisper", "caption"] else no_distributor
        called_from_slurm = "GLOBAL_RANK" in os.environ
    elif config["distribution"]["distributor"] == "pyspark":
        distributor_fn = pyspark_distributor
    elif config["distribution"]["distributor"] == "slurm":
        worker_args = {key: local_args[key] for key in local_args if not key.startswith("slurm")}
        slurm_args = config["distribution"]["distributor_args"]

        distributor_fn = SlurmDistributor(worker_args=worker_args, **slurm_args)
    else:
        raise ValueError(f"Distributor {config['distribution']['distributor']} not supported")

    distributor_fn(
        config["distribution"]["processes_count"],
        worker,
        shard_iterator,
        config["distribution"]["subjob_size"],
        max_shard_retry,
    )
    logger_process.join()
    if not called_from_slurm:
        fs.rm(run_tmp_dir, recursive=True)


def main():
    # Adjust the start and end indices as needed
    start_index = 2
    end_index = 7151  # Inclusive
    input_folder = '.'  # Folder containing your CSV files
    output_folder = "/mnt/data/victor/data/valleytest"

    # Loop over all CSV files
    for i in range(start_index, end_index + 1):
        csv_file = "".join(["/mnt/data/victor/data/webvid-10M/data/train/partitions/", f"{i:04d}.csv"])
        if not os.path.exists(csv_file):
            print(f'CSV file {csv_file} does not exist. Skipping.')
            continue
        print(f'Processing {csv_file}')
        custom_video2dataset(
            url_list=csv_file,
            output_folder=output_folder,
            output_format='files',
            input_format='csv',
            url_col='contentUrl',
            caption_col='name',
            save_additional_columns=['videoid', 'page_dir'],
            custom_sample_writer_class=CustomSampleWriter,
            config='default',  # Use 'default' config or specify your custom config
        )


if __name__ == '__main__':
    main()
