import argparse
import json
import os
from tqdm import tqdm
import zipfile

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Zip long clips.')
    parser.add_argument('video_folder', type=str, help='Input video folder')
    parser.add_argument('input_file', type=str, help='Input JOSNL file')
    parser.add_argument('output_folder', type=str, help='Output folder')
    parser.add_argument('--min_duration', type=float, default=10, help='Minimum duration of clips to be zipped')
    parser.add_argument('--max_duration', type=float, default=20, help='Maximum duration of clips to be zipped')
    parser.add_argument('--max_num_clips', type=int, default=1000, help='Maximum number of clips to be zipped')
    args = parser.parse_args()

    dic_list = []
    for line in open(args.input_file, 'r'):
        dic_list.append(json.loads(line))
    os.makedirs(args.output_folder, exist_ok=True)
    output_zip_file = os.path.join(args.output_folder, 'long_clips.zip')
    output_jsonl_file = os.path.join(args.output_folder, 'long_clips.jsonl')
    clip_count = 0
    with zipfile.ZipFile(output_zip_file, 'w') as zipf, open(output_jsonl_file, 'w') as out_file:
        for dic in tqdm(dic_list, desc='Zipping clips'):
            if clip_count >= args.max_num_clips:
                break
            video_path = os.path.join(args.video_folder, dic['path'])
            if dic['duration'] > args.min_duration and dic['duration'] < args.max_duration:
                zipf.write(video_path, os.path.basename(video_path))
                out_file.write(json.dumps(dic) + '\n')
                out_file.flush()
                clip_count += 1



if __name__ == '__main__':
    main()