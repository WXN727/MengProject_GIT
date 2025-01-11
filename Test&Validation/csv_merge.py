import os
import pandas as pd
import re

# Directory containing the CSV files
directory_path = '/home/wxn/Documents/allcsv'

# Output file path
output_file = '/home/wxn/Documents/merged_file.csv'


# Custom function for natural sorting of filenames
def natural_sort_key(filename):
    # Extract numerical components for natural sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]


# Create a list to store the DataFrames
dataframes = []

# Check if the directory exists
if os.path.exists(directory_path):
    # Get a sorted list of CSV files (natural sorting)
    csv_files = sorted([file for file in os.listdir(directory_path) if file.endswith('.csv')],
                       key=natural_sort_key)

    # Read each CSV file and append its DataFrame
    for file in csv_files:
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate all DataFrames into one
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Sort the merged DataFrame by the `filename` column in natural order
    if 'filename' in merged_df.columns:
        merged_df = merged_df.sort_values(by='filename', key=lambda col: col.map(natural_sort_key))

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"CSV files have been merged and saved to: {output_file}")
else:
    print(f"Directory does not exist: {directory_path}")
