import os
import cv2
from datetime import datetime

def create_output_directory(base_path, dir_suffix=""):
    current_time = datetime.now().strftime("%Y_%m_%d_%H:%M")
    i = 1
    dir_name = f"{current_time}_observation_{i}{dir_suffix}"
    while os.path.exists(os.path.join(base_path, dir_name)):
        i += 1
        dir_name = f"{current_time}_observation_{i}{dir_suffix}"
    output_path = os.path.join(base_path, dir_name)
    os.makedirs(output_path)
    return output_path

def add_timestamp(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    text_size = cv2.getTextSize(timestamp, font, 0.5, 1)[0]
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = frame.shape[0] - 10
    cv2.putText(frame, timestamp, (text_x, text_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def record_continuous_video(output_path, duration=None):
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))  # Retrieve and store frame rate

    video_name = os.path.join(output_path, "continuous_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Switch to MJPG codec
    out = cv2.VideoWriter(video_name, fourcc, frame_rate, (frame_width, frame_height))

    start_time = datetime.now()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = add_timestamp(frame)
        out.write(frame)

        if duration:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if elapsed_time >= duration:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return video_name, frame_rate  # Return video path and frame rate used

def split_video(video_path, output_dir, clip_duration, frame_rate):
    cap = cv2.VideoCapture(video_path)
    frames_per_clip = frame_rate * clip_duration  # Use passed frame rate

    clip_count = 1
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPG codec

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        clip_name = os.path.join(output_dir, f"clip_{clip_count:04d}.avi")
        out = cv2.VideoWriter(clip_name, fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))

        for _ in range(frames_per_clip):
            if not ret:
                break
            frame = add_timestamp(frame)
            out.write(frame)
            ret, frame = cap.read()

        out.release()
        clip_count += 1

    cap.release()

if __name__ == "__main__":
    base_input_path = '/home/wxn/PycharmProjects/MENGProject/input'

    # Create output directory for continuous video recording
    new_dir = create_output_directory(base_input_path)
    print(f"Created output directory: {new_dir}")

    # Record a continuous video for a specified duration (e.g., 100 seconds)
    try:
        continuous_video_path, frame_rate = record_continuous_video(new_dir, duration=100)
        print(f"Continuous video recorded at: {continuous_video_path}")

        # Create directories for different clip durations
        one_sec_dir = create_output_directory(new_dir, "_1sec_clips")
        two_sec_dir = create_output_directory(new_dir, "_2sec_clips")
        five_sec_dir = create_output_directory(new_dir, "_5sec_clips")
        ten_sec_dir = create_output_directory(new_dir, "_10sec_clips")

        # Split continuous video into clips of different durations, using consistent frame rate
        split_video(continuous_video_path, one_sec_dir, clip_duration=1, frame_rate=frame_rate)
        split_video(continuous_video_path, two_sec_dir, clip_duration=2, frame_rate=frame_rate)
        split_video(continuous_video_path, five_sec_dir, clip_duration=5, frame_rate=frame_rate)
        split_video(continuous_video_path, ten_sec_dir, clip_duration=10, frame_rate=frame_rate)

        print("Video split into clips of 1, 2, 5, and 10 seconds successfully.")

    except KeyboardInterrupt:
        print("Process terminated by user.")
