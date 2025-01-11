
import os
import cv2

def split_video(video_path, output_dir, clip_length=5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count // fps

    for start_time in range(0, duration, clip_length):
        output_path = os.path.join(output_dir, f"clip_{start_time:03d}.mp4")
        os.system(f"ffmpeg -i {video_path} -ss {start_time} -t {clip_length} -c copy {output_path}")
    cap.release()
# Example usage
split_video("/home/wxn/PycharmProjects/MENGProject/testing_videos/actions/walk.mp4",
            "/home/wxn/PycharmProjects/MENGProject/testing_videos/actions/walk_clips/", clip_length=5)
