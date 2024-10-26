'''
import os
from tqdm import tqdm
from video2dataset.main import video2dataset
from video2dataset.data_writer import FilesSampleWriter
from video2dataset.types import EncodeFormats
from typing import Optional, List, Any

# Custom sample writer to save videos to 'valley/PAGE_DIR/VIDEOID'
class CustomSampleWriter(FilesSampleWriter):
    def write(self, sample, shard_id, sample_id):
        # Extract page_dir and videoid from sample metadata
        page_dir = sample['meta'].get('page_dir', 'unknown')
        videoid = sample['meta'].get('videoid', 'unknown')

        # Construct the output directory and file path
        output_dir = os.path.join(self.output_folder, str(page_dir))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, str(videoid))

        # Save the video
        if 'video' in sample:
            with open(output_path, 'wb') as f:
                f.write(sample['video'])

        # Optionally, save the caption
        if 'caption' in sample['meta']:
            with open(output_path + '.txt', 'w') as f:
                f.write(sample['meta']['caption'])

# Modified video2dataset function to accept sample_writer_class
def video2dataset(
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
    max_shard_retry: int = 1,
    tmp_dir: str = "/tmp",
    config: Any = "default",
    sample_writer_class=None,
):
    """
    [Original docstring of video2dataset]
    """
    import os
    import sys
    import signal
    import fsspec
    from omegaconf import OmegaConf
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

    def identity(x):
        return x

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

    # TODO: find better location for this code
    # TODO: figure out minimum yt_meta_args for subtitles to be added to metadata
    if config["storage"]["captions_are_subtitles"]:
        assert clip_col is None  # no weird double-clipping
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

    save_caption = caption_col is not None or config["storage"]["captions_are_subtitles"]

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

    if sample_writer_class is None:
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
    output_folder = "/mnt/data/victor/data/valleytest"
    url_col = "contentUrl"
    caption_col = "name"
    save_additional_columns = ["videoid", "page_dir"]
    encode_formats = {'video': 'mp4'}

    num_csv_files = 7152  # Adjust as needed
    csv_files = ["".join(["/mnt/data/victor/data/webvid-10M/data/train/partitions", f"{i:04d}.csv"]) for i in range(num_csv_files)]

    for csv_file in tqdm(csv_files):
        video2dataset(
            url_list=csv_file,
            output_folder=output_folder,
            output_format='files',
            input_format='csv',
            url_col=url_col,
            caption_col=caption_col,
            save_additional_columns=save_additional_columns,
            encode_formats=encode_formats,
            stage='download',
            config='default',
            sample_writer_class=CustomSampleWriter,  # Pass the custom sample writer here
        )

if __name__ == "__main__":
    main()



import os
from tqdm import tqdm
from typing import Optional, List, Any
from video2dataset.main import video2dataset as original_video2dataset
from video2dataset.data_writer import FilesSampleWriter
from video2dataset.types import EncodeFormats

# Custom sample writer to save videos to 'valley/PAGE_DIR/VIDEOID'
class CustomSampleWriter(FilesSampleWriter):
    def write(self, sample, shard_id, sample_id):
        # Extract page_dir and videoid from sample metadata
        page_dir = sample['meta'].get('page_dir', 'unknown')
        videoid = sample['meta'].get('videoid', 'unknown')

        # Construct the output directory and file path
        output_dir = os.path.join(self.output_folder, str(page_dir))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, str(videoid))

        # Save the video
        if 'video' in sample:
            with open(output_path, 'wb') as f:
                f.write(sample['video'])

        # Optionally, save the caption
        if 'caption' in sample['meta']:
            with open(output_path + '.txt', 'w') as f:
                f.write(sample['meta']['caption'])

# Modified video2dataset function to accept sample_writer_class
def video2dataset(
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
    max_shard_retry: int = 1,
    tmp_dir: str = "/tmp",
    config: Any = "default",
    sample_writer_class=None,
):
    """
    [Original docstring of video2dataset]
    """
    import sys
    import signal
    import fsspec
    from omegaconf import OmegaConf
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

    def identity(x):
        return x

    # Prepare local_args without unpicklable objects
    local_args = {
        'url_list': url_list,
        'output_folder': output_folder,
        'output_format': output_format,
        'input_format': input_format,
        'encode_formats': encode_formats,
        'stage': stage,
        'url_col': url_col,
        'caption_col': caption_col,
        'clip_col': clip_col,
        'save_additional_columns': save_additional_columns,
        'enable_wandb': enable_wandb,
        'wandb_project': wandb_project,
        'incremental_mode': incremental_mode,
        'max_shard_retry': max_shard_retry,
        'tmp_dir': tmp_dir,
        'config': config,
        # Exclude 'sample_writer_class' to avoid pickling issues
    }

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

    # TODO: find better location for this code
    # TODO: figure out minimum yt_meta_args for subtitles to be added to metadata
    if config["storage"]["captions_are_subtitles"]:
        assert clip_col is None  # no weird double-clipping
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

    save_caption = caption_col is not None or config["storage"]["captions_are_subtitles"]

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

    if sample_writer_class is None:
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

    print(f"Starting the processing of file: {url_list}")
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
    output_folder = "/mnt/data/victor/data/valleytest"
    url_col = "contentUrl"
    caption_col = "name"
    save_additional_columns = ["videoid", "page_dir"]
    encode_formats = {'video': 'mp4'}

    num_csv_files = 7152  # Adjust as needed
    csv_files = ["".join(["/mnt/data/victor/data/webvid-10M/data/train/partitions/", f"{i:04d}.csv"]) for i in range(num_csv_files)]

    for csv_file in tqdm(csv_files):
        video2dataset(
            url_list=csv_file,
            output_folder=output_folder,
            output_format='files',
            input_format='csv',
            url_col=url_col,
            caption_col=caption_col,
            save_additional_columns=save_additional_columns,
            encode_formats=encode_formats,
            stage='download',
            config='default',
            sample_writer_class=CustomSampleWriter,  # Pass the custom sample writer here
        )

if __name__ == "__main__":
    main()
'''


import os
from tqdm import tqdm
from typing import Optional, List, Any
from video2dataset.main import video2dataset as original_video2dataset
from video2dataset.data_writer import FilesSampleWriter
from video2dataset.types import EncodeFormats

# Custom sample writer to save videos to 'valley/PAGE_DIR/VIDEOID'
class CustomSampleWriter(FilesSampleWriter):
    def write(self, key, sample, shard_id, sample_id):
        # Extract page_dir and videoid from sample metadata
        page_dir = sample['meta'].get('page_dir', 'unknown')
        videoid = sample['meta'].get('videoid', 'unknown')

        # Construct the output directory and file path
        output_dir = os.path.join(self.output_folder, str(page_dir))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, str(videoid))

        # Save the video
        if 'video' in sample:
            with open(output_path, 'wb') as f:
                f.write(sample['video'])

        # Optionally, save the caption
        if 'caption' in sample['meta']:
            with open(output_path + '.txt', 'w') as f:
                f.write(sample['meta']['caption'])

# Move identity function to the module level to make it picklable
def identity(x):
    return x

# Modified video2dataset function to accept sample_writer_class
def video2dataset(
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
    max_shard_retry: int = 1,
    tmp_dir: str = "/tmp",
    config: Any = "default",
    sample_writer_class=None,
):
    """
    [Original docstring of video2dataset]
    """
    import sys
    import signal
    import fsspec
    from omegaconf import OmegaConf
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

    # Prepare local_args without unpicklable objects
    local_args = {
        'url_list': url_list,
        'output_folder': output_folder,
        'output_format': output_format,
        'input_format': input_format,
        'encode_formats': encode_formats,
        'stage': stage,
        'url_col': url_col,
        'caption_col': caption_col,
        'clip_col': clip_col,
        'save_additional_columns': save_additional_columns,
        'enable_wandb': enable_wandb,
        'wandb_project': wandb_project,
        'incremental_mode': incremental_mode,
        'max_shard_retry': max_shard_retry,
        'tmp_dir': tmp_dir,
        'config': config,
        # Exclude 'sample_writer_class' to avoid pickling issues
    }

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

    # TODO: find better location for this code
    # TODO: figure out minimum yt_meta_args for subtitles to be added to metadata
    if config["storage"]["captions_are_subtitles"]:
        assert clip_col is None  # no weird double-clipping
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

    save_caption = caption_col is not None or config["storage"]["captions_are_subtitles"]

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

    if sample_writer_class is None:
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

    print(f"Starting the processing of file: {url_list}")
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
    output_folder = "/mnt/data/victor/data/valleytest"
    url_col = "contentUrl"
    caption_col = "name"
    save_additional_columns = ["videoid", "page_dir"]
    encode_formats = {'video': 'mp4'}

    num_csv_files = 7152  # Adjust as needed
    csv_files = ["".join(["/mnt/data/victor/data/webvid-10M/data/train/partitions/", f"{i:04d}.csv"]) for i in range(num_csv_files)]

    for csv_file in tqdm(csv_files[:1]):
        print(csv_file)
        video2dataset(
            url_list=csv_file,
            output_folder=output_folder,
            output_format='files',
            input_format='csv',
            url_col=url_col,
            caption_col=caption_col,
            save_additional_columns=save_additional_columns,
            encode_formats=encode_formats,
            stage='download',
            config='default',
            sample_writer_class=CustomSampleWriter,  # Pass the custom sample writer here
        )

if __name__ == "__main__":
    main()
