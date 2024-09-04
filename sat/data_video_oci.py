import os
import json
from typing import MutableSequence
import random
import decord
from torch.utils.data import Dataset
from decord import VideoReader
import numpy as np
import torch
import torchvision.transforms as TT
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr

def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if len(tensor) < num_frames:
        pad_length = num_frames - len(tensor)
        # Use the last frame to pad instead of zero
        last_frame = tensor[-1]
        pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]



class SftJsonlDataset(Dataset):
    def __init__(self, video_folder, jsonl_paths, video_size, fps, max_num_frames, skip_frms_num=3, text_key="text"):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(SftJsonlDataset, self).__init__()
        self.video_folder = video_folder
        self.metadata_list = []
        if not isinstance(jsonl_paths, MutableSequence):
            jsonl_paths = [jsonl_paths]
        skipped_videos = 0
        for jsonl_path in jsonl_paths:
            with open(jsonl_path, "r") as f:
                for line in f:
                    metadata = json.loads(line)
                    # duration = metadata["duration"]
                    # num_frames = metadata["num_frames"]
                    # actual_fps = num_frames / duration
                    # required_duration = max_num_frames / fps + 2 * skip_frms_num / actual_fps
                    # if duration < required_duration:
                    #     # print(f"Skipping video {metadata['path']} because its duration is too short, {duration} < {required_duration}")
                    #     skipped_videos += 1
                    #     continue

                    self.metadata_list.append(metadata)

        print(f"Skipped {skipped_videos} videos")
        print(f"Loaded {len(self.metadata_list)} videos")
        decord.bridge.set_bridge("torch")
        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num
        self.text_key = text_key

    def __getitem__(self, index):
        for i in range(10):
            try:
                return self.load_video_by_index(index)
            except TimeoutError as e:
                print(f"Error loading video {index}, retrying")
                print(e)
                index = random.randint(0, len(self.metadata_list) - 1)

    # def load_video_by_index(self, index):
    #     metadata = self.metadata_list[index]
    #     video_path = os.path.join(self.video_folder, metadata["path"])
    #     duration = metadata["duration"]
    #     num_frames = metadata["num_frames"]
    #     actual_fps = num_frames / duration
    #     with open(video_path, "rb") as f:
    #         frames = process_video(
    #             f,
    #             num_frames=self.max_num_frames,
    #             wanted_fps=self.fps,
    #             image_size=self.video_size,
    #             duration=duration,
    #             actual_fps=actual_fps,
    #             skip_frms_num=self.skip_frms_num,
    #         )
    #         frames = (frames - 127.5) / 127.5

    #     item = {
    #         "mp4": frames,
    #         "txt": metadata[self.text_key],
    #         "num_frames": self.max_num_frames,
    #         "fps": self.fps,
    #     }
    #     return item
    def load_video_by_index(self, index):
        # decord.bridge.set_bridge("torch")

        metadata = self.metadata_list[index]
        video_path = os.path.join(self.video_folder, metadata["path"])
        vr = VideoReader(uri=video_path, height=-1, width=-1)
        actual_fps = vr.get_avg_fps()
        ori_vlen = len(vr)

        if ori_vlen / actual_fps * self.fps > self.max_num_frames:
            num_frames = self.max_num_frames
            start = int(self.skip_frms_num)
            end = int(start + num_frames / self.fps * actual_fps)
            end_safty = min(int(start + num_frames / self.fps * actual_fps), int(ori_vlen))
            indices = np.arange(start, end, (end - start) // num_frames).astype(int)
            temp_frms = vr.get_batch(np.arange(start, end_safty))
            assert temp_frms is not None
            tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
            tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
        else:
            if ori_vlen > self.max_num_frames:
                num_frames = self.max_num_frames
                start = int(self.skip_frms_num)
                end = int(ori_vlen - self.skip_frms_num)
                indices = np.arange(start, end, (end - start) // num_frames).astype(int)
                temp_frms = vr.get_batch(np.arange(start, end))
                assert temp_frms is not None
                tensor_frms = (
                    torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
                )
                tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
            else:

                def nearest_smaller_4k_plus_1(n):
                    remainder = n % 4
                    if remainder == 0:
                        return n - 3
                    else:
                        return n - remainder + 1

                start = int(self.skip_frms_num)
                end = int(ori_vlen - self.skip_frms_num)
                num_frames = nearest_smaller_4k_plus_1(
                    end - start
                )  # 3D VAE requires the number of frames to be 4k+1
                end = int(start + num_frames)
                temp_frms = vr.get_batch(np.arange(start, end))
                assert temp_frms is not None
                tensor_frms = (
                    torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
                )

        tensor_frms = pad_last_frame(
            tensor_frms, self.max_num_frames
        )  # the len of indices may be less than num_frames, due to round error
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        tensor_frms = resize_for_rectangle_crop(tensor_frms, self.video_size, reshape_mode="center")
        tensor_frms = (tensor_frms - 127.5) / 127.5

        item = {
            "mp4": tensor_frms,
            "txt": metadata[self.text_key],
            "num_frames": num_frames,
            "fps": self.fps,
        }
        return item

    def __len__(self):
        return len(self.metadata_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(jsonl_paths=path, **kwargs)
