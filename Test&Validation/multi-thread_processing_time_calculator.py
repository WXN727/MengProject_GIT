import pandas as pd

# Load the data from the CSV file
file_path = '/home/wxn/PycharmProjects/MengProject_GIT/Test&Validation/2thread_results/10sec/2nd_processing_time_analysis_2thread_10sec_11_04_10:11.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Add a column to indicate pair index (e.g., 0, 1, 0, 1 for consecutive pairs)
data['pair_index'] = data.index // 2



# Group by 'pair_index' and calculate the maximum 'total_time' within each group
max_processing_times = data.groupby('pair_index')['total_time'].max().reset_index(name='max_processing_time')

# Display or save the result
print(max_processing_times)

# Optional: Save the result to a new CSV file
output_path = 'max_processing_times.csv'  # Replace with desired output file path
max_processing_times.to_csv(output_path, index=False)
