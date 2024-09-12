import os
import subprocess
import json

def get_video_duration(file_path):
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
        return duration / 3600  # Convert seconds to hours
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 0

def count_videos_and_duration(folder_path):
    total_videos = 0
    total_hours = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more video extensions if needed
                total_videos += 1
                video_path = os.path.join(root, file)
                duration = get_video_duration(video_path)
                total_hours += duration

    return total_videos, total_hours

if __name__ == "__main__":
    folder_path = "data/tandj/videos/"  # Replace with the actual folder path
    videos, hours = count_videos_and_duration(folder_path)
    print(f"Total number of videos: {videos}")
    print(f"Total duration: {hours:.2f} hours")

