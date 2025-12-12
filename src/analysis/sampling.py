#!/usr/bin/env python3
import os
import random
import subprocess
import tempfile
import shutil
import sys
from urllib.parse import urlparse, unquote

def check_dependencies():
    """Checks if ffmpeg and ffprobe are available in the system."""
    required = ['ffmpeg', 'ffprobe']
    missing = [cmd for cmd in required if shutil.which(cmd) is None]
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}. Please install them before running this script.")
        sys.exit(1)

def get_video_name_from_url(url):
    """
    Extracts the video name from a URL
    """
    parsed = urlparse(url)
    path = unquote(parsed.path)
    name = os.path.basename(path)
    video_name, _ = os.path.splitext(name)
    return video_name if video_name else None

def get_scene_change_timestamps(video_path):
    """
    Detects scene changes in a video using ffmpeg's scdet filter.
    Returns a sorted list of timestamps (in seconds) where scene changes occur,
    or None if no scene changes are detected.
    """
    fps = get_fps(video_path)

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "scdet=threshold=1.0",  # Adjust threshold as needed
        "-f", "null",
        "-"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    timestamps = []
    for line in result.stderr.splitlines():
        if "lavfi.scd.time" in line:
            try:
                time_part = line.split("lavfi.scd.time:")[1].split()[0]
                pts_time = float(time_part)
                frame_index = int(round(pts_time * fps))
                timestamps.append((pts_time, frame_index))
            except (IndexError, ValueError):
                continue

    formatted_timestamps = [f"{ts:.6f} (frame {idx})" for ts, idx in timestamps]
    print(f"Parsed timestamps: {formatted_timestamps}")

    return sorted([ts for ts, _ in timestamps]) if timestamps else None

def get_total_frames(video_path):
    """
    Counts the total number of frames in a video.
    Returns the frame count as an integer.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not result.stdout.strip().isdigit():
        raise Exception(f"Could not get frame count for {video_path}")
    return int(result.stdout.strip())

def get_fps(video_path):
    """
    Retrieves the frame rate (fps) for the given video.
    Returns fps as a float.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise Exception(f"Could not get FPS for {video_path}")
    fps_str = result.stdout.strip()
    try:
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
    except Exception as e:
        raise Exception(f"Error parsing FPS for {video_path}: {e}")
    return fps

def download_video(url, temp_dir):
    """
    Downloads a video from a URL to a temporary file.
    Returns the path to the temporary file.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=temp_dir)
    cmd = [
        "ffmpeg",
        "-i", url,
        "-c", "copy",  # Copy without re-encoding
        "-y",  # Overwrite if exists
        temp_file.name
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise Exception(f"Failed to download video from {url}: {result.stderr}")
    return temp_file.name

def process_video(video_path, video_name, output_folder, desired_frames):
    """
    Process a single video (local or downloaded) and extract frames.
    """
    fps = get_fps(video_path)
    scenes = get_scene_change_timestamps(video_path)

    print("Processing scene data, please wait...")

    total_frames = get_total_frames(video_path)

    if scenes:
        print(f"Detected {len(scenes)} scene changes in {video_name}")
    else:
        print(f"No scene changes detected in {video_name}")

    num_sampled_frames = min(desired_frames, total_frames)
    sampled_points = []

    if scenes:
        if len(scenes) >= num_sampled_frames:
            sampled_points = sorted(random.sample(scenes, num_sampled_frames))
        else:
            sampled_points = scenes.copy()
            remaining = num_sampled_frames - len(scenes)
            random_samples = random.sample(range(total_frames), remaining)
            sampled_points.extend(random_samples)
            sampled_points = sorted(sampled_points)
    else:
        sampled_points = sorted(random.sample(range(total_frames), num_sampled_frames))

    for point in sampled_points:
        if isinstance(point, float):
            frame_index = int(round(point * fps))
            cmd = [
                "ffmpeg",
                "-ss", str(point),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                os.path.join(output_folder, f"{video_name}_frame{frame_index}.jpg")
            ]
        else:
            frame_index = point
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vf", f"select='eq(n\\,{frame_index})'",
                "-vsync", "vfr",
                "-q:v", "2",
                os.path.join(output_folder, f"{video_name}_frame{frame_index}.jpg")
            ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error extracting frame {frame_index} from {video_name}: {result.stderr}")
        else:
            print(f"Extracted frame {frame_index} to folder: {output_folder}")
    print(f"Finished processing {video_name}. Extracted {num_sampled_frames} frames.\n")

def main():
    check_dependencies()
    # delete and re-create frames folder on each execution
    frames_dir = os.path.join(os.getcwd(), "frames")
    if os.path.exists(frames_dir):
        try:
            shutil.rmtree(frames_dir)
        except Exception as e:
            print(f"Error deleting existing frames directory: {e}")
    os.makedirs(frames_dir, exist_ok=True)

    mode = input("Choose 'remote' or 'local': ").strip().lower()
    if mode not in ['remote', 'local']:
        print("Invalid choice. Please enter 'remote' or 'local'.")
        return

    if mode == 'remote':
        try:
            num_videos = int(input("Enter the number of videos to process: "))
            if num_videos <= 0:
                raise ValueError("Number of videos must be positive.")
        except ValueError:
            print("Invalid number of videos.")
            return

        video_urls = []
        for i in range(num_videos):
            url = input(f"Enter the URL for video {i + 1}: ").strip()
            video_urls.append(url)

        try:
            desired_frames = int(input("Enter the desired number of frames to sample from each video: "))
            if desired_frames <= 0:
                raise ValueError("Number of frames must be positive.")
        except ValueError:
            print("Invalid number of frames.")
            return

        temp_dir = tempfile.mkdtemp()
        try:
            for i, url in enumerate(video_urls):
                print(f"\nDownloading video from {url}")
                temp_video_path = download_video(url, temp_dir)
                video_name = get_video_name_from_url(url)
                if not video_name:
                    video_name = f"video_{i + 1}"
                output_folder = os.path.join(frames_dir, video_name)
                os.makedirs(output_folder, exist_ok=True)
                process_video(temp_video_path, video_name, output_folder, desired_frames)
                os.remove(temp_video_path)  # Clean up temporary file after processing
        finally:
            # Ensure temporary directory is removed even if interrupted
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    elif mode == 'local':
        video_folder = input("Enter the path to the folder where videos are stored: ").strip()
        if not os.path.isdir(video_folder):
            print("The provided path is not a valid directory.")
            return

        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
        video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder)
                       if f.lower().endswith(video_extensions)]
        if not video_files:
            print("No video files found in the specified folder.")
            return

        try:
            desired_frames = int(input("Enter the desired number of frames to sample from each video: "))
            if desired_frames <= 0:
                raise ValueError("Number of frames must be positive.")
        except ValueError:
            print("Invalid number of frames.")
            return

        for video in video_files:
            video_name = os.path.splitext(os.path.basename(video))[0]
            output_folder = os.path.join(frames_dir, video_name)
            os.makedirs(output_folder, exist_ok=True)
            print(f"\nProcessing video: {video}")
            process_video(video, video_name, output_folder, desired_frames)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up temporary directories and exiting.")
        sys.exit(1)