import os

# Path to your directory
directory = "/home/wxn/Documents/allcsv"

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):  # Only process CSV files
        file_path = os.path.join(directory, filename)

        # Read the file
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Replace all single quotes
        updated_content = content.replace("'", "")

        # Write the updated content back to the file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(updated_content)

print("All single quotes have been removed from CSV files.")
