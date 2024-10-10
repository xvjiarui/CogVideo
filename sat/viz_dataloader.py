import os
import random

import torch
import torchvision.io as io
from data_video_oci import SftJsonlDatasetV2 as _SftJsonlDatasetV2

# Set a fixed seed for reproducibility
random.seed(42)

class SftJsonlDatasetV2(_SftJsonlDatasetV2):
    pass

dataset = SftJsonlDatasetV2(
    jsonl_paths=["data/tandj/metadata/g6s/tandj_recap.jsonl"],
    video_folder="data/tandj/clips_g6s/",
    text_key="text",
    video_size=[480, 720],
    fps=8,
    filter_metadata=False,
    max_num_frames=49,
    skip_frms_num=0.0,
)

# Create output directory if it doesn't exist
output_dir = "output/debug/dataset/clips_g6s"
os.makedirs(output_dir, exist_ok=True)


# Get a random list of indices
num_samples = 10  # You can adjust this number as needed
random_indices = random.sample(range(len(dataset)), num_samples)

print(f"Random indices: {random_indices}")


for i in random_indices:
    item = dataset[i]
    metadata = dataset.metadata_list[i]

    # Extract video tensor and metadata
    video_tensor = item["mp4"]
    num_frames = item["num_frames"]
    fps = item["fps"]
    text = item["txt"]

    # Denormalize the video tensor
    video_tensor = (video_tensor * 127.5 + 127.5).byte()

    # Ensure video tensor is in the correct format [T, C, H, W]
    video_tensor = video_tensor.permute(0, 2, 3, 1)  # Change to [T, H, W, C] for saving

    # Save the video
    output_path = os.path.join(output_dir, f"video_{i:04d}.mp4")
    io.write_video(output_path, video_tensor[:num_frames], fps=fps)

    print(metadata)
    print(f"Saved video {i} to {output_path}")
    import ipdb; ipdb.set_trace()
    pass
