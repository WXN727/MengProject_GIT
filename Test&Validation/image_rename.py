import os

# Define the root directory where images are stored
root_dir = "/home/wxn/PycharmProjects/MENGProject/detectron2_repo/img/original"

# Walk through the subfolders and sub-subfolders of the root directory
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)

    if os.path.isdir(subfolder_path):
        for sub_subfolder in os.listdir(subfolder_path):
            sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)

            if os.path.isdir(sub_subfolder_path):
                # Use only the sub-subfolder name as the prefix
                folder_name = os.path.basename(sub_subfolder_path)

                # Rename each image file in the sub-subfolder
                for file in os.listdir(sub_subfolder_path):
                    if file.startswith("frame_") and file.endswith(('.png', '.jpg', '.jpeg')):
                        # Construct old and new file paths
                        old_file_path = os.path.join(sub_subfolder_path, file)
                        new_file_name = f"{folder_name}_{file}"
                        new_file_path = os.path.join(sub_subfolder_path, new_file_name)

                        # Rename the file in its original location
                        os.rename(old_file_path, new_file_path)

print("All images have been renamed in their original directories.")
