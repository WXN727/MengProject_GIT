import os

# Path to the directory containing the videos
directory = "/home/wxn/PycharmProjects/MENGProject/testing_videos/actions/walk_clips"

# Ensure the directory exists
if os.path.exists(directory):
    # Get a list of all files in the directory
    files = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

    # Rename each file
    for idx, file in enumerate(files, start=1):
        # Extract the file extension
        file_extension = os.path.splitext(file)[1]
        # Construct the new file name
        new_name = f"walk_clip_{idx}{file_extension}"
        # Full paths for renaming
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)
        # Rename the file
        os.rename(old_path, new_path)

    print(f"Renaming complete. Total files renamed: {len(files)}.")
else:
    print(f"The directory '{directory}' does not exist. Please check the path.")
