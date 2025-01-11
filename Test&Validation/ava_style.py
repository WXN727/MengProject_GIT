import pandas as pd
import ast
import natsort

# Frame dimensions
frame_width = 1280
frame_height = 720

# Load the input CSV
input_csv_path = "/home/wxn/Documents/new_csv_sorted.csv"  # Replace with your input CSV file
output_csv_path = "/home/wxn/Documents/ava_style_annotations.csv"  # Corrected output file

# Read the CSV
df = pd.read_csv(input_csv_path)

# Correct the `video_id` extraction
def extract_video_id(filename):
    parts = filename.split('_')
    clip_index = parts.index("clip")  # Find the "clip" part
    return '_'.join(parts[:clip_index + 2])  # Include "clip" and its number

df['video_id'] = df['filename'].apply(extract_video_id)

# Extract frame_sec from the filename
df['frame_sec'] = df['filename'].apply(lambda x: int(x.split('_')[-1].split('.')[0]))

# Parse bounding box details and normalize
def parse_bbox(bbox_str):
    bbox = ast.literal_eval(bbox_str)
    x1 = bbox['x'] / frame_width
    y1 = bbox['y'] / frame_height
    x2 = (bbox['x'] + bbox['width']) / frame_width
    y2 = (bbox['y'] + bbox['height']) / frame_height
    return x1, y1, x2, y2

df[['x1', 'y1', 'x2', 'y2']] = df['region_shape_attributes'].apply(parse_bbox).apply(pd.Series)

# Add action IDs
action_labels = {
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
df['action_id'] = df['video_id'].apply(lambda x: next((v for k, v in action_labels.items() if k in x), None))

# Use natural sorting for proper order
df = df.sort_values(
    by=['video_id', 'frame_sec'],
    key=lambda col: natsort.natsort_keygen()(col)
)

# Select relevant columns
ava_df = df[['video_id', 'frame_sec', 'x1', 'y1', 'x2', 'y2', 'action_id']]

# Save the corrected and normalized AVA-style file
ava_df.to_csv(output_csv_path, index=False, header=False)

print(f"Corrected and normalized AVA-style annotations saved to: {output_csv_path}")
