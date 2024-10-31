
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

## Step By Step file 
- Step 1: `update_labels.py`- Script to copy corresponding annotation files (.xml) for each image in the training, validation, and testing folders,
 ensuring that each image in these folders has its matching annotation from the main annotations folder.
- Step 2:`convert_labels_to_single_lego_label.py` Script for unifying labels,let all the labels in the xml files be 'lego'
- Step 3:`dataset_converter.py`- convert the dataset from VOC format to YOLO format,  Prepares data structure required for YOLOv5 training
- Step 4: Training Script (in YOLOv5 repository & local Canva file):
  - `train.py`: Main training script for YOLO model
  - Located in cloned YOLOv5 repository， available in local Canva file.
  - Handles model training with specified parameters
- `requirements.txt`: List of project dependencies

## Requirements
- Python 3.x
- PyTorch
- YOLOv5
- All additional dependencies are listed in \`requirements.txt\`


# Setup and Usage
# 1. Clone YOLOv5 repository
cd /Users/sophiawang/Desktop/Lab3
git clone https://github.com/ultralytics/yolov5

# 2. Create and activate a virtual environment
python3 -m venv lego_env
source lego_env/bin/activate

# 3.Clone YOLOv5 (required, not included in this repo due to size):
git clone https://github.com/ultralytics/yolov5
cd yolov5

# 4.Install YOLOv5 dependencies
cd yolov5
pip install -r requirements.txt

# 5. Install additional required packages
## Install-YOLOv5-dependencies
pip install -r requirements.txt

## Install-additional-required-packages
pip install pandas matplotlib opencv-python pillow PyYAML tqdm torch torchvision seaborn requests

# 6. Evaluate the model-training
python3 train.py --img 640 --batch 16 --epochs 100 --data ../yolo_dataset/data.yaml --weights yolov5s.pt --name lego_detector

#### Evaluation Metrics Explained
- **mAP@0.5**: Mean Average Precision at an IoU threshold of 0.5
- **Precision**: Measures the accuracy of positive predictions
- **Recall**: Measures the completeness of the model in finding positive samples
- **F1 Score**: Harmonic mean of Precision and Recall, providing a single metric for performance


