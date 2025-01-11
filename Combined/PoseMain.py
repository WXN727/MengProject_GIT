import subprocess
import os
import cv2
from datetime import datetime
import threading
import queue


def create_output_directory(base_path):
    current_time = datetime.now().strftime("%Y_%m_%d_%H:%M")
    i = 1
    dir_name = f"{current_time}_observation_{i}"
    while os.path.exists(os.path.join(base_path, dir_name)):
        i += 1
        dir_name = f"{current_time}_observation_{i}"
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


def record_video(output_path, duration=1, video_queue=None):
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))
    frame_count = frame_rate * duration

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_count = 1

    while cap.isOpened():
        video_name = os.path.join(output_path, f"video_{video_count:04d}.avi")
        out = cv2.VideoWriter(video_name, fourcc, frame_rate, (frame_width, frame_height))

        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frame = add_timestamp(frame)
            out.write(frame)

        out.release()
        video_count += 1

        if video_queue:
            video_queue.put(video_name)

    cap.release()
    cv2.destroyAllWindows()


def run_yolo_slowfast():
    command = "python yolo_slowfast_threading.py"
    try:
        process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.stdout.decode()
        print("yolo_slowfast.py executed successfully:")
        print(output)
    except subprocess.CalledProcessError as e:
        print("Error occurred while running yolo_slowfast.py:")
        print(e.stderr.decode())


if __name__ == "__main__":
    base_input_path = '/home/wxn/PycharmProjects/MENGProject/input'
    base_output_path = '/home/wxn/PycharmProjects/MENGProject/output'

    new_dir = create_output_directory(base_input_path)
    print(f"Created input directory: {new_dir}")

    video_queue = queue.Queue()

    recorder_thread = threading.Thread(target=record_video, args=(new_dir, 1, video_queue))
    recorder_thread.start()

    try:
        run_yolo_slowfast()  # Call yolo_slowfast.py once while videos are being captured
    except KeyboardInterrupt:
        print("Terminated by user")
    finally:
        video_queue.put(None)  # Send exit signal to the recorder thread if needed

    recorder_thread.join()
