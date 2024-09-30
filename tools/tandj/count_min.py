import argparse
import os
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def get_video_info(file_path):
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        file_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        duration = float(data['format']['duration'])
        # Try to get nb_frames, if not available, calculate from duration and fps
        video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
        if video_stream and 'nb_frames' in video_stream:
            frames = int(video_stream['nb_frames'])
        elif video_stream and 'r_frame_rate' in video_stream:
            fps = eval(video_stream['r_frame_rate'])
            frames = int(duration * fps)
        else:
            frames = 0
        
        return duration / 3600, frames  # Convert duration to hours
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 0, 0

def process_video(file_path):
    hours, frames = get_video_info(file_path)
    return hours, frames

def count_videos_and_info(folder_path):
    video_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more video extensions if needed
                video_files.append(os.path.join(root, file))

    total_videos = len(video_files)
    total_hours = 0
    total_frames = 0

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_video, file_path) for file_path in video_files]
        for future in tqdm(futures, desc="Processing videos"):
            hours, frames = future.result()
            print(frames)
            total_hours += hours
            total_frames += frames

    return total_videos, total_hours, total_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count videos, their total duration, and total frames in a folder.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing videos")
    args = parser.parse_args()
    folder_path = args.folder_path
    videos, hours, frames = count_videos_and_info(folder_path)
    print(f"Total number of videos: {videos}")
    print(f"Total duration: {hours:.2f} hours")
    print(f"Total number of frames: {frames}")
    avg_fps = frames / hours / 3600
    print(f"Average FPS: {avg_fps:.2f}")
    avg_frames_per_video = frames / videos
    print(f"Average frames per video: {avg_frames_per_video:.2f}")
    avg_length_per_video = hours / videos * 3600
    print(f"Average length per video: {avg_length_per_video:.2f} seconds")

