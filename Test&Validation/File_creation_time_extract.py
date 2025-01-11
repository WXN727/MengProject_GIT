import os
import pandas as pd
from datetime import datetime
import re

# Directories for processed and observation files
processed_dir = '/home/wxn/PycharmProjects/MENGProject/output/2024_10_29_00:39_processed_1'
observation_dir = '/home/wxn/PycharmProjects/MENGProject/input/2024_10_29_00:39_observation_1'

# Output Excel file path
output_excel_file = '/home/wxn/PycharmProjects/MENGProject/Test&Validation/slowfast_latency_check_10_29_00:39.xlsx'


# Function to extract numeric part from file names for proper sorting
def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else -1


def get_creation_times(directory):
    # Create a dictionary to store file names and creation times
    file_times = {}

    # Loop through the files in the directory
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        if os.path.isfile(file_path):
            # Get the file creation time
            creation_time = os.path.getctime(file_path)
            creation_time_formatted = datetime.fromtimestamp(creation_time)
            # Store in dictionary as {file_name: creation_time}
            file_times[file_name] = creation_time_formatted

    return file_times


def calculate_time_difference(processed_times, observation_times):
    # Create a list to store the file name, processed time, observation time, and time difference
    data = []

    for file_name in sorted(processed_times.keys(), key=extract_number):
        if file_name in observation_times:
            processed_time = processed_times[file_name]
            observation_time = observation_times[file_name]
            # Calculate time difference in seconds
            time_diff = (processed_time - observation_time).total_seconds()
        else:
            processed_time = processed_times[file_name]
            observation_time = None
            time_diff = None

        data.append([file_name, processed_time, observation_time, time_diff])

    return data


if __name__ == "__main__":
    # Get creation times for both directories
    processed_times = get_creation_times(processed_dir)
    observation_times = get_creation_times(observation_dir)

    # Calculate the time difference between corresponding files
    data = calculate_time_difference(processed_times, observation_times)

    # Convert the data into a DataFrame and save it to an Excel file
    df = pd.DataFrame(data, columns=['File Name', 'Processed Creation Time', 'Observation Creation Time',
                                     'Time Difference (seconds)'])

    # Save the DataFrame to an Excel file
    df.to_excel(output_excel_file, index=False)
    print(f"File creation times and time differences saved to {output_excel_file}")
