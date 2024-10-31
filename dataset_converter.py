# step 3: convert the dataset from VOC format to YOLO format

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random
import logging
from tqdm import tqdm

class DatasetConverter:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dataset_conversion.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def convert_bbox_voc_to_yolo(self, size, box):
        """Convert VOC format bounding box to YOLO format"""
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        
        # Convert VOC format (xmin, ymin, xmax, ymax) to YOLO format (x_center, y_center, width, height)
        x = (box[0] + box[2]) / 2.0
        y = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        # Normalize
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        
        return (x, y, w, h)

    def create_dataset_structure(self):
        """Create the directory structure for the YOLO dataset"""
        # Create main directories
        dataset_path = self.base_path / 'yolo_dataset'
        for split in ['train', 'val', 'test']:
            (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        return dataset_path

    def convert_annotation(self, xml_path, output_path):
        """Convert a single XML annotation to YOLO format"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            with open(output_path, 'w') as out_file:
                for obj in root.iter('object'):
                    # Always use class index 0 for 'lego'
                    class_id = 0
                    
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # Convert to YOLO format
                    bb = self.convert_bbox_voc_to_yolo((width, height), (xmin, ymin, xmax, ymax))
                    out_file.write(f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
            
            return True
        except Exception as e:
            self.logger.error(f"Error converting {xml_path}: {str(e)}")
            return False

    def split_dataset(self, image_paths):
        """Split dataset into train, validation, and test sets"""
        random.shuffle(image_paths)
        total = len(image_paths)
        
        train_split = int(0.7 * total)
        val_split = int(0.85 * total)
        
        return {
            'train': image_paths[:train_split],
            'val': image_paths[train_split:val_split],
            'test': image_paths[val_split:]
        }

    def copy_and_convert_dataset(self):
        """Copy images and convert annotations to YOLO format"""
        # Create dataset structure
        dataset_path = self.create_dataset_structure()
        
        # Get all image paths
        image_paths = list(self.base_path.glob('**/*.jpg'))
        splits = self.split_dataset(image_paths)
        
        # Process each split
        for split_name, split_images in splits.items():
            self.logger.info(f"Processing {split_name} split ({len(split_images)} images)")
            
            for img_path in tqdm(split_images, desc=f"Processing {split_name}"):
                # Define paths
                xml_path = img_path.with_suffix('.xml')
                if not xml_path.exists():
                    xml_path = self.base_path / 'annotations' / xml_path.name.replace('.jpg', '.xml')
                
                if not xml_path.exists():
                    self.logger.warning(f"No annotation found for {img_path}")
                    continue
                
                # Copy image
                dest_img = dataset_path / split_name / 'images' / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Convert and save annotation
                label_path = dataset_path / split_name / 'labels' / img_path.with_suffix('.txt').name
                if not self.convert_annotation(xml_path, label_path):
                    self.logger.warning(f"Failed to convert annotation for {img_path}")
                    continue

        return dataset_path

    def create_data_yaml(self, dataset_path):
        """Create data.yaml file for YOLOv5"""
        yaml_content = f"""
# Train/val/test paths
train: {dataset_path}/train/images
val: {dataset_path}/val/images
test: {dataset_path}/test/images

# number of classes
nc: 1

# class names
names: ['lego']
"""
        yaml_path = dataset_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content.strip())
        
        return yaml_path

    def process_dataset(self):
        """Main function to process the entire dataset"""
        self.logger.info("Starting dataset conversion")
        dataset_path = self.copy_and_convert_dataset()
        yaml_path = self.create_data_yaml(dataset_path)
        self.logger.info(f"Dataset conversion completed. YAML file created at {yaml_path}")
        return dataset_path, yaml_path

if __name__ == "__main__":
    # Initialize converter
    converter = DatasetConverter('/Users/sophiawang/Desktop/Lab3')
    
    # Process dataset
    dataset_path, yaml_path = converter.process_dataset()