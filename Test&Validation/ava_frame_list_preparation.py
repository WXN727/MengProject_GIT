from natsort import natsorted
import os
import pandas as pd

# Paths
frames_dir = "/home/wxn/PycharmProjects/MENGProject/dataset/frames"
output_train_csv = "/home/wxn/PycharmProjects/MENGProject/dataset/frame_lists/train.csv"
output_val_csv = "/home/wxn/PycharmProjects/MENGProject/dataset/frame_lists/val.csv"

# Action label mapping
action_mapping = {
    "bend": 1,
    "crawl": 2,
    "crouch_kneel": 3,
    "fall_down": 4,
    "fight": 5,
    "lie": 6,
    "sit": 7,
    "stand": 8,
    "walk": 9,
}

# Define the range of clip numbers for train and validation splits
def is_train_clip(video_id):
    clip_number = int(video_id.split("_clip_")[-1])
    return 1 <= clip_number <= 5

def is_val_clip(video_id):
    clip_number = int(video_id.split("_clip_")[-1])
    return 6 <= clip_number <= 10

# Collect frame information with action labels
def collect_frames(frames_dir, clip_filter):
    data = []
    for root, _, files in os.walk(frames_dir):
        for file in files:
            if file.endswith(".jpg"):
                video_id = os.path.basename(root)  # e.g., 30fps_bend_clip_1
                if clip_filter(video_id):  # Check if the video belongs to the split
                    frame_id = int(file.split("_")[-1].split(".")[0])  # Extract frame number
                    path = os.path.join(root, file).replace(frames_dir + "/", "")  # Relative path
                    action_name = video_id.split("_clip_")[0].split("_", 1)[-1]  # Extract action name (e.g., bend)
                    label = action_mapping.get(action_name, "")  # Map action to label ID
                    data.append([video_id, video_id, frame_id, path, label])
    # Sort the data using natural sorting
    data = natsorted(data, key=lambda x: (x[0], x[2]))
    return data

# Generate train and val frame lists
train_data = collect_frames(frames_dir, is_train_clip)
val_data = collect_frames(frames_dir, is_val_clip)

# Validate rows
for row in train_data:
    if len(row) != 5:
        print(f"Invalid train row: {row}")
for row in val_data:
    if len(row) != 5:
        print(f"Invalid val row: {row}")

# Save to CSV
columns = ["original_video_id", "video_id", "frame_id", "path", "labels"]
pd.DataFrame(train_data, columns=columns).to_csv(output_train_csv, index=False, header=False)
pd.DataFrame(val_data, columns=columns).to_csv(output_val_csv, index=False, header=False)

print(f"Train frame list saved to: {output_train_csv}")
print(f"Validation frame list saved to: {output_val_csv}")
