import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Plot the stats of the dataset.')
    parser.add_argument('input_file', type=str, help='Input JSONL file')
    args = parser.parse_args()

    dic_list = []
    for line in open(args.input_file, 'r'):
        dic = json.loads(line)
        if dic['num_frames'] > 3000:
            continue
        dic_list.append(dic)
    
    # Number of frames histogram
    num_frames = [dic['num_frames'] for dic in dic_list]
    # Print histogram data for number of frames
    print("Number of frames histogram:")
    hist, bin_edges = np.histogram(num_frames, bins=20)
    for i in range(len(hist)):
        print(f"Bin {i+1}: {bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}, Count: {hist[i]}")

    print("\nNumber of frames statistics:")
    print(f"Min: {min(num_frames)}")
    print(f"Max: {max(num_frames)}")
    print(f"Mean: {np.mean(num_frames):.2f}")
    print(f"Median: {np.median(num_frames):.2f}")
    print(f"Standard deviation: {np.std(num_frames):.2f}")
    plt.hist(num_frames, bins=20)
    plt.title('Number of frames histogram')
    plt.xlabel('Number of frames')
    plt.ylabel('Number of videos')
    plt.xticks(np.linspace(min(num_frames), max(num_frames), 10))  # Set 20 x-ticks
    output_dir = os.path.dirname(args.input_file)
    input_file_name = os.path.splitext(os.path.basename(args.input_file))[0]
    plt.savefig(os.path.join(output_dir, f'{input_file_name}_num_frames_hist.png'))
    plt.clf()

    # Duration histogram
    durations = [dic['duration'] for dic in dic_list]

    # Print histogram data for duration
    print("\nDuration histogram:")
    hist, bin_edges = np.histogram(durations, bins=20)
    for i in range(len(hist)):
        print(f"Bin {i+1}: {bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}, Count: {hist[i]}")

    print("\nDuration statistics:")
    print(f"Min: {min(durations):.2f}")
    print(f"Max: {max(durations):.2f}")
    print(f"Mean: {np.mean(durations):.2f}")
    print(f"Median: {np.median(durations):.2f}")
    print(f"Standard deviation: {np.std(durations):.2f}")

    plt.hist(durations, bins=20)
    plt.title('Duration histogram')
    plt.xlabel('Duration')
    plt.ylabel('Number of videos')
    plt.xticks(np.linspace(min(durations), max(durations), 10))  # Set 20 x-ticks
    plt.savefig(os.path.join(output_dir, f'{input_file_name}_duration_hist.png'))


if __name__ == '__main__':
    main()