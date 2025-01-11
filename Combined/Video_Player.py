import os
import cv2

# Path to the directory where videos are stored
video_directory = '/home/wxn/PycharmProjects/MENGProject/output/2024_10_29_01:03_processed_1'


def video_play(directory):
    # Get all video file names in the directory (sorted to play in order)
    video_files = sorted([f for f in os.listdir(directory) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])

    if not video_files:
        print("No video files found in the directory!")
        return

    # Open the same window for continuous video playback
    cv2.namedWindow('CCTV Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CCTV Feed', 640, 480)  # Adjust window size as needed

    for video_file in video_files:
        video_path = os.path.join(directory, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop when video ends

            # Display the frame in the same window
            cv2.imshow('CCTV Feed', frame)

            # Wait for 25ms between frames (adjust to match frame rate)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return  # Exit if 'q' is pressed

        cap.release()  # Release video capture for the current video

    # Close the window after all videos have been played
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_play(video_directory)
