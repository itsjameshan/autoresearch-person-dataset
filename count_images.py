
import os

# Define the paths
dataset_path = r"D:\PythonProject\person_dataset"
split = "train"

def count_label_files(directory):
    """Count the number of label files (.txt) in a directory (excluding classes.txt)."""
    count = 0
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".txt") and filename != "classes.txt":
                count += 1
    return count

label_dir = os.path.join(dataset_path, "labels", split)
num = count_label_files(label_dir)
print(f"Number of images in {split} split (based on labels): {num}")
