import pandas as pd

# Load the full dataset (AVA-style annotations)
input_csv_path = "/home/wxn/Documents/ava_style_annotations.csv"  # Replace with your AVA-style annotation file
train_output_csv = "/home/wxn/Documents/ava_train_annotations.csv"
val_output_csv = "/home/wxn/Documents/ava_val_annotations.csv"

# Read the dataset
df = pd.read_csv(input_csv_path, header=None)
df.columns = ["video_id", "frame_sec", "x1", "y1", "x2", "y2", "action_id"]

# Define train and validation splits
def split_dataset(video_id):
    clip_number = int(video_id.split("_clip_")[-1])  # Extract the clip number
    if 1 <= clip_number <= 5:  # Clips 1-5 for training
        return "train"
    elif 6 <= clip_number <= 10:  # Clips 6-10 for validation
        return "val"
    else:
        return None

# Apply the split logic
df["split"] = df["video_id"].apply(split_dataset)

# Save train and validation datasets
train_df = df[df["split"] == "train"].drop(columns=["split"])
val_df = df[df["split"] == "val"].drop(columns=["split"])

train_df.to_csv(train_output_csv, index=False, header=False)
val_df.to_csv(val_output_csv, index=False, header=False)

print(f"Training annotations saved to: {train_output_csv}")
print(f"Validation annotations saved to: {val_output_csv}")
