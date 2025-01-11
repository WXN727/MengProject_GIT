import cv2
import os
import time

def get_hour_slot():
    current_time = time.localtime()
    start_hour = current_time.tm_hour
    start_time = time.strftime(f"%Y%m%d_%H:00", current_time)
    end_hour = (start_hour + 1) % 24
    end_time = time.strftime(f"%Y%m%d_", time.localtime(time.mktime(current_time) + 3600)) + f"{end_hour:02}:00"
    return f"{start_time}-{end_time}"

def capture_video(base_dir, video_length=10, frame_rate=30):
    while True:
        # Create the folder name based on the current hour slot
        folder_name = get_hour_slot()
        output_dir = os.path.join(base_dir, folder_name)

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize the webcam
        cap = cv2.VideoCapture(0)

        try:
            while True:
                # Get the current timestamp for the video filename
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                video_filename = os.path.join(output_dir, f"video_{timestamp}.avi")

                # Get the default width and height of the frame
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(video_filename, fourcc, frame_rate, (frame_width, frame_height))

                # Calculate the number of frames to capture
                num_frames = video_length * frame_rate

                print("Recording video... Press 'Ctrl+C' to stop.")

                # Capture frames from the webcam
                for _ in range(num_frames):
                    ret, frame = cap.read()
                    if ret:
                        out.write(frame)
                    else:
                        break

                # Release the file writer after each 10-second video
                out.release()
                print(f"Video saved as {video_filename}")

        except KeyboardInterrupt:
            # Handle the user pressing Ctrl+C
            print("Recording stopped.")
            break

        finally:
            # Release the webcam
            cap.release()

if __name__ == "__main__":
    base_directory = "/home/wxn/PycharmProjects/MENGProject/input"  # Change this to your desired base directory
    capture_video(base_directory)
