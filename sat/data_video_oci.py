import os
import json
import threading
import random
import decord
from torch.utils.data import Dataset
from decord import VideoReader
import numpy as np
import torch
from data_video import resize_for_rectangle_crop, pad_last_frame, process_video

def load_video_with_timeout(video_path, timeout=20):
    video_container = {}

    def target_function():
        video = VideoReader(uri=video_path, height=-1, width=-1)
        video_container["video"] = video

    thread = threading.Thread(target=target_function)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("Loading video timed out")
        raise TimeoutError
    return video_container.get("video", None)



class SftJsonlDataset(Dataset):
    def __init__(self, video_folder, jsonl_paths, video_size, fps, max_num_frames, skip_frms_num=3, text_key="text"):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(SftJsonlDataset, self).__init__()
        self.video_folder = video_folder
        self.metadata_list = []
        if isinstance(jsonl_paths, str):
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
            except (TimeoutError, RuntimeError) as e:
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
        # vr = VideoReader(uri=video_path, height=-1, width=-1)
        vr = load_video_with_timeout(video_path)
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

class SftJsonlDatasetV2(Dataset):
    def __init__(
        self,
        video_folder,
        jsonl_paths,
        video_size,
        fps,
        max_num_frames,
        skip_frms_num=3,
        text_key="text",
        sub_text_key="cap",
        filter_metadata=True,
        ):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(SftJsonlDatasetV2, self).__init__()
        self.video_folder = video_folder
        self.metadata_list = []
        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]
        short_videos = 0
        long_videos = 0
        for jsonl_path in jsonl_paths:
            with open(jsonl_path, "r") as f:
                for line in f:
                    metadata = json.loads(line)
                    duration = metadata["duration"]
                    num_frames = metadata["num_frames"]
                    actual_fps = num_frames / duration
                    required_duration = max_num_frames / fps + 2 * skip_frms_num / actual_fps
                    if duration < required_duration:
                        short_videos += 1
                        if filter_metadata:
                            continue
                    if num_frames > 8000:
                        long_videos += 1
                        if filter_metadata:
                            continue

                    self.metadata_list.append(metadata)

        print('filter_metadata:', filter_metadata)
        print(f"{'Skipped' if filter_metadata else 'Loaded'} {short_videos} short videos")
        print(f"{'Skipped' if filter_metadata else 'Loaded'} {long_videos} long videos")
        print(f'Loaded {len(self.metadata_list)} videos')
        decord.bridge.set_bridge("torch")
        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num
        self.text_key = text_key
        self.sub_text_key = sub_text_key
        self.filter_metadata = filter_metadata
        
    def __getitem__(self, index):
        for i in range(10):
            try:
                if self.filter_metadata:
                    return self.load_video_by_index_after_filter(index)
                else:
                    return self.load_video_by_index(index)
            except (TimeoutError, RuntimeError) as e:
                print(f"Error loading video {index}, retrying")
                print(e)
                index = random.randint(0, len(self.metadata_list) - 1)
    
    def load_txt_by_index(self, index):
        metadata = self.metadata_list[index]
        cap_data = metadata[self.text_key]
        if isinstance(cap_data, list):
            num_caps = len(cap_data)
            cap_idx = random.randint(0, num_caps - 1)
            return cap_data[cap_idx][self.sub_text_key]
        else:
            return cap_data

    def load_video_by_index_after_filter(self, index):
        metadata = self.metadata_list[index]
        video_path = os.path.join(self.video_folder, metadata["path"])
        duration = metadata["duration"]
        num_frames = metadata["num_frames"]
        actual_fps = num_frames / duration
        with open(video_path, "rb") as f:
            frames = process_video(
                f,
                num_frames=self.max_num_frames,
                wanted_fps=self.fps,
                image_size=self.video_size,
                duration=duration,
                actual_fps=actual_fps,
                skip_frms_num=self.skip_frms_num,
            )
            frames = (frames - 127.5) / 127.5

        item = {
            "mp4": frames,
            "txt": self.load_txt_by_index(index),
            "num_frames": self.max_num_frames,
            "fps": self.fps,
        }
        return item

    def load_video_by_index(self, index):
        # decord.bridge.set_bridge("torch")

        metadata = self.metadata_list[index]
        video_path = os.path.join(self.video_folder, metadata["path"])
        # vr = VideoReader(uri=video_path, height=-1, width=-1)
        vr = load_video_with_timeout(video_path)
        actual_fps = vr.get_avg_fps()
        ori_vlen = len(vr)

        if ori_vlen / actual_fps * self.fps > self.max_num_frames:
            num_frames = self.max_num_frames
            max_seek = int(ori_vlen - self.skip_frms_num - num_frames / self.fps * actual_fps)
            start = random.randint(self.skip_frms_num, max_seek + 1)
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
                indices = np.arange(start, end, max((end - start) // num_frames, 1)).astype(int)
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
            "txt": self.load_txt_by_index(index),
            "num_frames": num_frames,
            "fps": self.fps,
        }
        return item

    def __len__(self):
        return len(self.metadata_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(jsonl_paths=path, **kwargs)
