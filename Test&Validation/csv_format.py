def clean_csv_format(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Replace commas with spaces (to match the blog's format)
            cleaned_line = line.replace(',', ' ')
            # Replace six double quotes with two double quotes
            cleaned_line = cleaned_line.replace('""""""', '""')  # Or replace with '' if needed
            outfile.write(cleaned_line)
    print(f"Cleaned and formatted file saved to: {output_file}")

# Paths to the CSV files
input_train_csv = "/home/wxn/PycharmProjects/MENGProject/dataset/frame_lists/train.csv"
input_val_csv = "/home/wxn/PycharmProjects/MENGProject/dataset/frame_lists/val.csv"
output_train_csv = "/home/wxn/Documents/modified_list_csv/formatted_train.csv"
output_val_csv = "/home/wxn/Documents/modified_list_csv/formatted_val.csv"

# Clean and format the train and val CSV files
clean_csv_format(input_train_csv, output_train_csv)
clean_csv_format(input_val_csv, output_val_csv)
