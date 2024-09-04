import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process CSV and video folder.')
    parser.add_argument('input_file', type=str, help='Input JSONL file')
    args = parser.parse_args()

    dic_list = []
    for line in open(args.input_file, 'r'):
        dic_list.append(json.loads(line))
    
    # Number of frames histogram
    num_frames = [dic['num_frames'] for dic in dic_list]
    plt.hist(num_frames, bins=20)
    plt.title('Number of frames histogram')
    plt.xlabel('Number of frames')
    plt.ylabel('Number of videos')
    plt.xticks(np.linspace(min(num_frames), max(num_frames), 10))  # Set 20 x-ticks
    output_dir = os.path.dirname(args.input_file)
    plt.savefig(os.path.join(output_dir, 'num_frames_hist.png'))
    plt.clf()

    # Duration histogram
    durations = [dic['duration'] for dic in dic_list]
    plt.hist(durations, bins=20)
    plt.title('Duration histogram')
    plt.xlabel('Duration')
    plt.ylabel('Number of videos')
    plt.xticks(np.linspace(min(durations), max(durations), 10))  # Set 20 x-ticks
    plt.savefig(os.path.join(output_dir, 'duration_hist.png'))


if __name__ == '__main__':
    main()