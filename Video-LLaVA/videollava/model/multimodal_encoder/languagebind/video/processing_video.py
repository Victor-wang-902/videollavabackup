
import torch
import cv2
import decord
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

decord.bridge.set_bridge('torch')

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def get_video_transform(config):
    config = config.vision_config
    if config.video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(config.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )

    elif config.video_decode_backend == 'decord':

        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif config.video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform


def load_and_transform_video(
        video_path,
        transform,
        video_decode_backend='opencv',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8,

):

    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'decord':
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(decord_vr)
        typee = 0
        if typee == 0: #normal (number at the end)
            frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        elif typee == 1: # only the beginning (no clear flag/ no number)
            frame_id_list = np.arange(num_frames, dtype=int)
        elif typee == 2: # reverse video (number in 1st frame)
            frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)[::-1]
        elif typee == 3: # beginning with flag/number inserted at 2nd
            frame_id_list = np.arange(num_frames, dtype=int)
            frame_id_list[1] = 1720
        elif typee == 4: # normal with flag/number inserted at 2nd
            frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
            frame_id_list[1] = 1720
        elif typee == 5: # beginning with flag/number inserted at 1st
            frame_id_list = np.arange(num_frames, dtype=int)
            frame_id_list[0] = 1720
        elif typee == 6: # normal with flag/number inserted at 1st
            frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
            frame_id_list[0] = 1720
        elif typee == 7: # beginning with flag/number inserted at end
            frame_id_list = np.arange(num_frames, dtype=int)
            frame_id_list[-1] = 1720
        elif typee == 8: # normal with flag/number inserted at the end
            frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
            frame_id_list[-1] = 1720

            

        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        
        
        video_outputs = transform(video_data)
        #video_outputs = np.array(video_data)
        #import imageio
        #import os

        # Directory to save the video
        #output_video_path = "output_video.mp4"
        #fps = 1  # Set frames per second for the output video

        # Ensure video_outputs has the shape (C, T, H, W)
        # Convert to (T, H, W, C) for saving as video
        #video_outputs_np = video_outputs.permute(1, 2, 3, 0).byte().cpu().numpy()  # Convert to (T, H, W, C)

        # Check that the video is in the expected shape (T, H, W, C)

        # Create a writer object for saving video
        #with imageio.get_writer(output_video_path, fps=fps, codec='libx264') as writer:
        #    for frame in video_outputs:
        #        frame_uint8 = frame.astype('uint8')  # Ensure the frame is in uint8 format for saving
        #        writer.append_data(frame_uint8)
        #raise Exception

        """import os
        from PIL import Image

        # Directory to save the frames
        save_dir = "/projectnb/ivc-ml/vwang/projects/attention_allocation/Video-LLaVA/videollava/eval/inference/extracted_frames"
        os.makedirs(save_dir, exist_ok=True)
        for idx, frame_id in enumerate(frame_id_list):
            frame = video_data[:, idx].permute(1, 2, 0).byte().cpu().numpy()  # Change shape to (H, W, C)
            frame_image = Image.fromarray(frame)
            frame_path = os.path.join(save_dir, f"frame_{frame_id}.jpg")
            frame_image.save(frame_path)
        raise Exception"""

    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            _, frame = cv2_vr.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
        video_outputs = transform(video_data)
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_outputs

class LanguageBindVideoProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindVideoTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_video_transform(config)
        self.image_processor = load_and_transform_video
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, context_length=77, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                      truncation=True, return_tensors=return_tensors, **kwargs)

        if images is not None:
            images = make_list_of_images(images)
            image_features = [self.image_processor(image, self.transform,
                                                   video_decode_backend=self.config.vision_config.video_decode_backend,
                                                   num_frames=self.config.vision_config.num_frames) for image in images]
            image_features = torch.stack(image_features)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {"pixel_values": image_features}

    def preprocess(self, images, return_tensors):
        return self.__call__(images=images, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
