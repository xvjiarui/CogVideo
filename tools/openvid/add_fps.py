import argparse
import json
import os
import subprocess
from tqdm import tqdm
from decord import VideoReader
from concurrent.futures import ThreadPoolExecutor

def get_video_info_ffprobe(video_path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,duration,nb_frames', '-of', 'json', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    video_info = json.loads(result.stdout)['streams'][0]
    
    width = int(video_info['width'])
    height = int(video_info['height'])
    duration = float(video_info['duration'])
    num_frames = int(video_info['nb_frames'])
    
    return num_frames, duration, width, height


def process_video(dic, video_folder):
    video_path = os.path.join(video_folder, dic['path'])
    # NOTE: decord gives too many open files error
    # vr = VideoReader(video_path)
    # dic['num_frames'] = len(vr)
    # dic['fps'] = vr.get_avg_fps()
    # next_frame = vr.next()
    # dic['height'] = next_frame.shape[0]
    # dic['width'] = next_frame.shape[1]
    num_frames, duration, width, height = get_video_info_ffprobe(video_path)
    dic['num_frames'] = num_frames
    dic['duration'] = duration
    dic['width'] = width
    dic['height'] = height

    return dic
    

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process CSV and video folder.')
    parser.add_argument('video_folder', type=str, help='Input video folder')
    parser.add_argument('input_file', type=str, help='Input JOSNL file')
    parser.add_argument('output_file', type=str, help='Output JOSNL file')
    args = parser.parse_args()

    dic_list = []
    for line in open(args.input_file, 'r'):
        dic_list.append(json.loads(line))
    
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_video, dic, args.video_folder) for dic in dic_list]
        with open(args.output_file, 'w') as out_file:
            for future in tqdm(futures, desc="Processing videos"):
                dic = future.result()
                out_file.write(json.dumps(dic) + '\n')


if __name__ == '__main__':
    main()