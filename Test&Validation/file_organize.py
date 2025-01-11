import os
import shutil

# Define the source and target directories
source_dir = "/home/wxn/PycharmProjects/MENGProject/detectron2_repo/img/detection"
target_dir = "/home/wxn/PycharmProjects/MENGProject/detectron2_repo/img/allcsv"

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Walk through the source directory
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".csv"):
            # Construct full file paths
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, file)

            # Copy the file to the target directory
            shutil.copy2(source_file, target_file)

print(f"All CSV files have been copied to {target_dir}.")
