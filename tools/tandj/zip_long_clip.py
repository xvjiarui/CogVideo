import argparse
import json
import os
from tqdm import tqdm
import zipfile
import pandas as pd
import re
def extract_video_id(video_name):
    # Extract episode ID from the video name using regular expression
    
    # Extract episode ID
    episode_id = None
    match = re.search(r' - (\d+) - ', video_name)
    if match:
        episode_id = int(match.group(1))
    
    # Extract scene ID
    scene_id = None
    match = re.search(r'_scene-(\d+)', video_name)
    if match:
        scene_id = int(match.group(1))
    
    # Combine episode ID and scene ID
    if episode_id is not None and scene_id is not None:
        combined_id = f"{episode_id:03d}_{scene_id:03d}"
    else:
        combined_id = None
    
    return combined_id

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Zip long clips.')
    parser.add_argument('video_folder', type=str, help='Input video folder')
    parser.add_argument('input_file', type=str, help='Input JOSNL file')
    parser.add_argument('output_folder', type=str, help='Output folder')
    parser.add_argument('--min_duration', type=float, default=8, help='Minimum duration of clips to be zipped')
    parser.add_argument('--max_duration', type=float, default=40, help='Maximum duration of clips to be zipped')
    parser.add_argument('--max_num_clips', type=int, default=5000, help='Maximum number of clips to be zipped')
    args = parser.parse_args()

    dic_list = []
    for line in open(args.input_file, 'r'):
        dic_list.append(json.loads(line))
    os.makedirs(args.output_folder, exist_ok=True)
    output_zip_file = os.path.join(args.output_folder, 'long_clips.zip')
    output_jsonl_file = os.path.join(args.output_folder, 'long_clips.jsonl')
    clip_count = 0
    data = []
    with zipfile.ZipFile(output_zip_file, 'w') as zipf, open(output_jsonl_file, 'w') as out_file:
        for dic in tqdm(dic_list, desc='Zipping clips'):
            if clip_count >= args.max_num_clips:
                break
            video_path = os.path.join(args.video_folder, dic['path'])
            video_id = extract_video_id(dic['path'])
            assert video_id is not None, f"Failed to extract video ID from {dic['path']}"
            if dic['duration'] > args.min_duration and dic['duration'] < args.max_duration:
                # zipf.write(video_path, os.path.basename(video_path))
                zipf.write(video_path, f"{video_id}.mp4")
                out_file.write(json.dumps(dic) + '\n')
                out_file.flush()
                clip_count += 1
    
                data.append({
                    'video name': f"{video_id}.mp4",
                    'original video name': os.path.basename(video_path),
                    'duration': dic['duration'],
                    'num frames': dic['num_frames']
                })

    print(f'Number of clips: {len(data)}')
    # Create a DataFrame using pandas
    df = pd.DataFrame(data)
    # Sort the DataFrame by 'video name'
    df = df.sort_values(by='video name')

    # Save the DataFrame to a CSV file
    csv_output_path = os.path.join(args.output_folder, 'long_clips_info.csv')
    df.to_csv(csv_output_path, index=False)

    print(f"CSV file with video information saved to: {csv_output_path}")



if __name__ == '__main__':
    main()