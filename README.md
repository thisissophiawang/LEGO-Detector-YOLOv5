
This project is for detecting LEGO pieces using a YOLOv5 model. It contains custom dataset configurations, training scripts, and validation results.

## Project Structure
- \`annotations/\`: Contains labeled data for the LEGO pieces.
- \`training/\`: Training data for the model.
- \`validation/\`: Validation data for model evaluation.
- \`yolo_dataset/\`: Dataset used for YOLOv5 training.

## Requirements
- Python 3.x
- PyTorch
- YOLOv5
- All additional dependencies are listed in \`requirements.txt\`


# Setup and Usage
## 1. Clone YOLOv5 repository
echo "Cloning YOLOv5 repository..."
cd /Users/sophiawang/Desktop/Lab3
git clone https://github.com/ultralytics/yolov5

# 2. Create and activate a virtual environment
echo "Creating virtual environment..."
python3 -m venv lego_env
source lego_env/bin/activate

# 3. Install YOLOv5 dependencies
echo "Installing YOLOv5 dependencies..."
cd yolov5
pip install -r requirements.txt

# 4. Install additional required packages
echo "Installing additional required packages..."
pip install pandas matplotlib opencv-python pillow PyYAML tqdm torch torchvision seaborn requests

# 5. Start training the YOLOv5 model
echo "Starting YOLOv5 training..."
python3 train.py --img 640 --batch 16 --epochs 100 --data ../yolo_dataset/data.yaml --weights yolov5s.pt --name lego_detector

# 6. Evaluate the model
echo "Evaluating the trained model..."
python3 val.py --weights runs/train/lego_detector/weights/best.pt --data ../yolo_dataset/data.yaml --iou-thres 0.5


#### Evaluation Metrics Explained
- **mAP@0.5**: Mean Average Precision at an IoU threshold of 0.5
- **Precision**: Measures the accuracy of positive predictions
- **Recall**: Measures the completeness of the model in finding positive samples
- **F1 Score**: Harmonic mean of Precision and Recall, providing a single metric for performance


