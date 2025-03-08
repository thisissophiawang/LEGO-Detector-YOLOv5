# LEGO-Detector-YOLOv5

This project implements LEGO piece detection using YOLOv5 model. It contains custom dataset configurations, training scripts, and validation results.

## Note about YOLOv5
The YOLOv5 repository is not included in this project due to its size. You will need to clone it separately during setup following the instructions below.
However, it's available in the local file which I posted in Canva.

## Project Structure
- `annotations/`: Contains labeled data for the LEGO pieces
- `training_set350/`: Training data (70% of dataset)
- `validation_set75/`: Validation data (15% of dataset)
- `testing_set75/`: Testing data (15% of dataset)
- `yolo_dataset/`: Processed dataset for YOLOv5 training

# HuggingFace Link:
https://huggingface.co/spaces/Thisissophia/3D-Image-Composer

# Visualizations Output:
🖼️ **YOLOv5 Validation Batch Example**  

![YOLOv5 Validation Batch](https://github.com/thisissophiawang/LEGO-Detector-YOLOv5/blob/main/Visualizations%20output%20/train/lego_detector/val_batch1_labels.jpg?raw=true)


## Step By Step file 
- **Step 1: `update_labels.py`**  
  - Script to copy corresponding annotation files (.xml) for each image in the training, validation, and testing folders, ensuring that each image in these folders has its matching annotation from the main annotations folder.

- **Step 2: `convert_labels_to_single_lego_label.py`**  
  - Script for unifying labels, setting all the labels in the XML files to 'lego'.

- **Step 3: `dataset_converter.py`**  
  - Converts the dataset from VOC format to YOLO format. Prepares the data structure required for YOLOv5 training.

- **Step 4: `train.py`   Training Script (in YOLOv5 repository & local Canva file)**  
  - `train.py`: Main training script for YOLO model.
  - Located in the cloned YOLOv5 repository, also available in the local Canva file.
  - Handles model training with specified parameters.
   
- `requirements.txt`: List of project dependencies

## Requirements
- Python 3.x
- PyTorch
- YOLOv5
- All additional dependencies are listed in \`requirements.txt\`


# Setup and Usage

## Clone initial repository
- `cd /Users/sophiawang/Desktop/Lab3`
- `git clone https://github.com/ultralytics/yolov5`

## Create and activate a virtual environment
- `python3 -m venv lego_env`
- `source lego_env/bin/activate`

## YOLOv5 setup, clone (required, not included in this repo due to size):
- `git clone https://github.com/ultralytics/yolov5`
- `cd yolov5`

## Install YOLOv5 dependencies
- `pip install -r requirements.txt`

## Install additional required packages
- `pip install -r requirements.txt`
- `pip install pandas matplotlib opencv-python pillow PyYAML tqdm torch torchvision seaborn requests`

## Evaluate the model-training
- `python3 train.py --img 640 --batch 16 --epochs 100 --data ../yolo_dataset/data.yaml --weights yolov5s.pt --name lego_detector`

## Training Command Parameters
- **`--img 640`**: Resizes input images to 640x640 pixels.
- **`--batch 16`**: Sets batch size to process 16 images at once.
- **`--epochs 100`**: Runs training for 100 complete passes through the dataset.
- **`--data ../yolo_dataset/data.yaml`**: Points to the dataset configuration file (paths and classes).
- **`--weights yolov5s.pt`**: Uses pre-trained YOLOv5 small model weights.
- **`--name lego_detector`**: Sets the name of the training experiment for easy tracking.

## Evaluation Metrics Explained
- **mAP@0.5**: Mean Average Precision at an IoU threshold of 0.5.
- **Precision**: Measures the accuracy of positive predictions.
- **Recall**: Measures the completeness of the model in finding positive samples.
- **F1 Score**: Harmonic mean of Precision and Recall, providing a single metric for performance


## Visualizations (under the local Canva file) or Visualizations output folder under github
Refer to:
Lab3/yolov5/runs/train/lego_detector  
Lab3/yolov5/runs/val/exp





