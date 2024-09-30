import argparse
import os

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from scenedetect import AdaptiveDetector, detect, FrameTimecode
from tqdm import tqdm

tqdm.pandas()


def process_single_row(row, min_scene_len, max_frames, adaptive_threshold):
    print(f"Processing video '{row['path']}' with min_scene_len={min_scene_len}, max_frames={max_frames}, adaptive_threshold={adaptive_threshold}")
    # windows
    # from scenedetect import detect, ContentDetector, AdaptiveDetector

    video_path = row["path"]

    detector = AdaptiveDetector(
        # adaptive_threshold=3.0,
        # adaptive_threshold=1.0,
        adaptive_threshold=adaptive_threshold,
        # adaptive_threshold=10.0,
        # min_scene_len=360,
        # min_scene_len=240,
        min_scene_len=min_scene_len,
        # luma_only=True,
    )
    # detector = ContentDetector()
    # TODO: catch error here
    try:
        scene_list = detect(video_path, detector, start_in_scene=False)
        
    except Exception as e:
        print(f"Video '{video_path}' with error {e}")
        return False, ""

    # Get the total duration of the video
    total_duration = scene_list[-1][1].get_seconds()
    
    # Filter out scenes completely within the first 30 seconds or last 10 seconds
    filtered_scene_list = [
        (s, t) for s, t in scene_list 
        if not (t.get_seconds() <= 30 or s.get_seconds() >= total_duration - 10)
    ]
    
    # Adjust the start and end times of the first and last scenes if they overlap with the excluded regions
    if filtered_scene_list:
        first_scene = filtered_scene_list[0]
        if first_scene[0].get_seconds() < 30:
            filtered_scene_list[0] = (FrameTimecode(float(30), fps=first_scene[0].framerate), first_scene[1])
        
        last_scene = filtered_scene_list[-1]
        if last_scene[1].get_seconds() > total_duration - 10:
            filtered_scene_list[-1] = (
                last_scene[0],
                FrameTimecode(float(total_duration - 10), fps=last_scene[1].framerate)
            )

    # for scene in filtered_scene_list:
    #     new_end_frame_idx = scene[1].get_frames()
    #     while (new_end_frame_idx-end_frame_idx[-1]) > (max_cutscene_len+2)*fps: # if no cutscene at min_scene_len+2, then cut at min_scene_len
    #         end_frame_idx.append(end_frame_idx[-1] + int(max_cutscene_len*fps))
    #     end_frame_idx.append(new_end_frame_idx)

    max_cutscene_len = max_frames / first_scene[0].framerate
    if max_cutscene_len:
        end_timecode = [filtered_scene_list[0][0].get_seconds()]
        for scene in filtered_scene_list:
            new_end_timecode = scene[1].get_seconds()
            while (new_end_timecode-end_timecode[-1]) >= (max_cutscene_len * 2): # if no cutscene at man_scene_len*1.5, then cut at min_scene_len
                end_timecode.append(end_timecode[-1] + max_cutscene_len)
            end_timecode.append(new_end_timecode)
        new_scene_list = []
        for i in range(len(end_timecode)-1):
            new_scene_list.append((FrameTimecode(end_timecode[i], fps=first_scene[0].framerate), FrameTimecode(end_timecode[i+1], fps=first_scene[0].framerate)))
        filtered_scene_list = new_scene_list
    
    timestamp = [(s.get_timecode(), t.get_timecode()) for s, t in filtered_scene_list]
    return True, str(timestamp)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None, help="#workers for pandarallel")
    parser.add_argument("--min_scene_len", type=int, default=15, help="min scene length")
    parser.add_argument("--max_frames", type=int, default=None, help="max number of frames")
    parser.add_argument("--threshold", type=float, default=3.0, help="threshold for adaptive detector")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    if args.num_workers is not None:
        pandarallel.initialize(progress_bar=True, nb_workers=args.num_workers)
    else:
        pandarallel.initialize(progress_bar=True)

    meta = pd.read_csv(meta_path)
    ret = meta.parallel_apply(process_single_row, axis=1, min_scene_len=args.min_scene_len, max_frames=args.max_frames, adaptive_threshold=args.threshold)

    succ, timestamps = list(zip(*ret))
    meta["timestamp"] = timestamps
    meta = meta[np.array(succ)]

    wo_ext, ext = os.path.splitext(meta_path)
    if args.out_path is None:
        out_path = f"{wo_ext}_timestamp{ext}"
    else:
        out_path = args.out_path
    meta.to_csv(out_path, index=False)
    print(f"New meta (shape={meta.shape}) with timestamp saved to '{out_path}'.")


if __name__ == "__main__":
    main()
