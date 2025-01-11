import os
import subprocess

# Specify the directory where your videos are stored
input_directory = '/home/wxn/PycharmProjects/MENGProject/testing_videos/actions/walk_clips/'  # Change this to your folder path
output_directory = '/home/wxn/PycharmProjects/MENGProject/testing_videos/actions/walk_clips/fixed_30fps'
# Loop through all the files in the directory
os.makedirs(output_directory, exist_ok=True)
for filename in os.listdir(input_directory):
    if filename.endswith(".mp4"):  # You can adjust the extension if needed
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, f"30fps_{filename}")

        # Construct the ffmpeg command
        command = [
            "ffmpeg",
            "-i", input_path,
            "-r", "30",
            output_path
        ]

        # Execute the command
        subprocess.run(command)
        print(f"Processed {filename}")
