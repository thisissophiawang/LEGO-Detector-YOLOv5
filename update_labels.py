# Step1: Use a Python script to batch modify tags in each XML markup file

# This script copies corresponding annotation files (.xml) for each image in the training, validation, and testing folders,
# ensuring that each image in these folders has its matching annotation from the main annotations folder.

import os
import shutil

# Define the path of the image folder and annotation folder
images_folders = {
    'training': '/Users/sophiawang/Desktop/Lab3/training set350',
    'validation': '/Users/sophiawang/Desktop/Lab3/Validation Set75',
    'testing': '/Users/sophiawang/Desktop/Lab3/Testing Set75'
}
annotations_folder = '/Users/sophiawang/Desktop/Lab3/annotations'

# Traverse each image folder, find the corresponding annotation file and copy it to the folder where the image is located
for set_type, folder_path in images_folders.items():
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            base_name = os.path.splitext(filename)[0]
            annotation_file = f"{base_name}.xml"
            annotation_path = os.path.join(annotations_folder, annotation_file)

            # Check if the annotation file exists and copy it to the folder where the image is located
            if os.path.exists(annotation_path):
                shutil.copy(annotation_path, folder_path)
            else:
                print(f"Annotation not found for image: {filename}")

print("The corresponding annotation files have been copied to the target folders.")
