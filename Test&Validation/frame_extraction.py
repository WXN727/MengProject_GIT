import cv2
import os
from glob import glob

# Function to extract frames at a specific rate
def extract_frames(video_path, output_dir, frame_rate=1):

    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # Debugging: Print FPS
    print(f"Detected FPS for {video_path}: {fps}")

    # Handle invalid FPS (0 or None)
    if fps is None or fps == 0:
        print(f"Warning: Could not determine FPS for {video_path}. Using default FPS = 30.")
        fps = 30  # Default to 30 FPS

    frame_interval = int(fps / frame_rate)  # Interval to save frames

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame based on the interval
        if frame_count % frame_interval == 0:
            frame_name = f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path} to {output_dir}")


# Function to process all videos in a directory
def batch_extract_frames(video_dir, output_dir, frame_rate=1):

    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    for video_name in video_files:
        video_path = os.path.join(video_dir, video_name)
        video_output_dir = os.path.join(output_dir, os.path.splitext(video_name)[0])
        os.makedirs(video_output_dir, exist_ok=True)

        print(f"Processing '{video_path}'...")
        extract_frames(video_path, video_output_dir, frame_rate)



video_directory = "/home/wxn/PycharmProjects/MENGProject/testing_videos/actions/walk_clips/fixed_30fps"
frames_directory = "/home/wxn/PycharmProjects/MENGProject/testing_videos/actions/walk_clips/1fps_walk_clips_frames/"
frames_directory_30 = "/home/wxn/PycharmProjects/MENGProject/testing_videos/actions/walk_clips/30fps_walk_clips_frames/"
# Extract frames at 1 FPS
batch_extract_frames(video_directory, frames_directory, frame_rate=1)

# Extract frames at 30 FPS (if needed, call again)
batch_extract_frames(video_directory, frames_directory_30, frame_rate=30)
