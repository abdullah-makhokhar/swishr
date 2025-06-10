"""
Dataset Manager for Basketball Detection
Handles dataset preparation, splitting, and validation
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple, List, Dict, Any
import yaml
import logging
from collections import Counter
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages basketball detection dataset preparation and validation"""
    
    def __init__(self, data_dir: str, output_dir: str = "data/basketball_model"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.classes = ['basketball', 'hoop', 'person']
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_dataset(self, train_split: float = 0.7, val_split: float = 0.2, 
                       test_split: float = 0.1, seed: int = 42) -> Dict[str, Any]:
        """
        Prepare dataset by splitting into train/val/test sets
        
        Args:
            train_split: Training set proportion
            val_split: Validation set proportion  
            test_split: Test set proportion
            seed: Random seed for reproducibility
            
        Returns:
            Dataset statistics
        """
        random.seed(seed)
        
        # Get all image files from training directory
        train_images_dir = self.data_dir / "train" / "images"
        train_labels_dir = self.data_dir / "train" / "labels"
        
        if not train_images_dir.exists():
            raise FileNotFoundError(f"Training images directory not found: {train_images_dir}")
            
        # Get all image files
        image_files = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))
        
        # Verify corresponding label files exist
        valid_pairs = []
        for img_file in image_files:
            label_file = train_labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                valid_pairs.append((img_file, label_file))
            else:
                logger.warning(f"No label file found for {img_file.name}")
                
        logger.info(f"Found {len(valid_pairs)} valid image-label pairs")
        
        # Shuffle and split
        random.shuffle(valid_pairs)
        n_total = len(valid_pairs)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_pairs = valid_pairs[:n_train]
        val_pairs = valid_pairs[n_train:n_train + n_val]
        test_pairs = valid_pairs[n_train + n_val:]
        
        # Create split directories
        splits = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
        
        dataset_stats = {}
        
        for split_name, pairs in splits.items():
            self._create_split_directory(split_name, pairs)
            stats = self._analyze_split(split_name, pairs)
            dataset_stats[split_name] = stats
            logger.info(f"{split_name.upper()} set: {len(pairs)} samples")
            
        # Create data.yaml for YOLO training
        self._create_data_yaml()
        
        # Generate dataset report
        self._generate_dataset_report(dataset_stats)
        
        return dataset_stats
        
    def _create_split_directory(self, split_name: str, pairs: List[Tuple[Path, Path]]):
        """Create directory structure for a dataset split"""
        split_dir = self.output_dir / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        # Create directories
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for img_file, label_file in pairs:
            # Copy image
            shutil.copy2(img_file, images_dir / img_file.name)
            # Copy and clean label file
            self._copy_clean_label(label_file, labels_dir / label_file.name)
            
    def _copy_clean_label(self, source_file: Path, dest_file: Path):
        """Copy label file and clean any formatting issues"""
        with open(source_file, 'r') as f:
            content = f.read().strip()
            
        # Remove trailing % character if present
        content = content.rstrip('%')
        
        # Clean up and validate each line
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 5:
                    try:
                        # Validate and reformat
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        if class_id < len(self.classes) and all(0 <= coord <= 1 for coord in coords):
                            cleaned_line = f"{class_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}"
                            cleaned_lines.append(cleaned_line)
                    except ValueError:
                        # Skip invalid lines
                        continue
                        
        # Write cleaned content
        with open(dest_file, 'w') as f:
            f.write('\n'.join(cleaned_lines))
            if cleaned_lines:  # Add newline at end if there are lines
                f.write('\n')
            
    def _analyze_split(self, split_name: str, pairs: List[Tuple[Path, Path]]) -> Dict[str, Any]:
        """Analyze dataset split for statistics"""
        class_counts = Counter()
        total_objects = 0
        image_sizes = []
        
        for img_file, label_file in pairs:
            # Read image to get dimensions
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w = img.shape[:2]
                image_sizes.append((w, h))
            
            # Read labels to count classes
            with open(label_file, 'r') as f:
                content = f.read().strip().rstrip('%')
                lines = content.split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            total_objects += 1
                            
        # Calculate statistics
        stats = {
            'num_images': len(pairs),
            'total_objects': total_objects,
            'class_distribution': dict(class_counts),
            'avg_objects_per_image': total_objects / len(pairs) if pairs else 0,
            'image_sizes': image_sizes
        }
        
        if image_sizes:
            widths, heights = zip(*image_sizes)
            stats['avg_width'] = np.mean(widths)
            stats['avg_height'] = np.mean(heights)
            stats['min_width'] = min(widths)
            stats['max_width'] = max(widths)
            stats['min_height'] = min(heights)
            stats['max_height'] = max(heights)
            
        return stats
        
    def _create_data_yaml(self):
        """Create data.yaml file for YOLO training"""
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': {i: name for i, name in enumerate(self.classes)}
        }
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
            
        logger.info(f"Created data.yaml at {yaml_path}")
        
    def _generate_dataset_report(self, stats: Dict[str, Any]):
        """Generate comprehensive dataset report"""
        report_path = self.output_dir / "dataset_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Basketball Dataset Report\\n\\n")
            f.write(f"Generated for training basketball detection model\\n\\n")
            
            for split_name, split_stats in stats.items():
                f.write(f"## {split_name.upper()} Set\\n\\n")
                f.write(f"- **Images**: {split_stats['num_images']}\\n")
                f.write(f"- **Total Objects**: {split_stats['total_objects']}\\n")
                f.write(f"- **Avg Objects/Image**: {split_stats['avg_objects_per_image']:.2f}\\n")
                
                if 'avg_width' in split_stats:
                    f.write(f"- **Avg Image Size**: {split_stats['avg_width']:.0f}x{split_stats['avg_height']:.0f}\\n")
                    
                f.write(f"\\n**Class Distribution**:\\n")
                for class_id, count in split_stats['class_distribution'].items():
                    class_name = self.classes[class_id] if class_id < len(self.classes) else f"unknown_{class_id}"
                    percentage = (count / split_stats['total_objects']) * 100 if split_stats['total_objects'] > 0 else 0
                    f.write(f"- {class_name}: {count} ({percentage:.1f}%)\\n")
                    
                f.write("\\n")
                
        logger.info(f"Generated dataset report at {report_path}")
        
    def validate_dataset(self) -> bool:
        """Validate the prepared dataset"""
        try:
            # Check if all required directories exist
            required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
            for dir_name in required_dirs:
                dir_path = self.output_dir / dir_name
                if not dir_path.exists():
                    logger.error(f"Required directory missing: {dir_path}")
                    return False
                    
            # Check if data.yaml exists
            yaml_path = self.output_dir / "data.yaml"
            if not yaml_path.exists():
                logger.error(f"data.yaml not found: {yaml_path}")
                return False
                
            # Validate a few samples
            train_images = list((self.output_dir / "train" / "images").glob("*"))
            if len(train_images) == 0:
                logger.error("No training images found")
                return False
                
            # Check first few image-label pairs
            for img_file in train_images[:5]:
                label_file = self.output_dir / "train" / "labels" / f"{img_file.stem}.txt"
                if not label_file.exists():
                    logger.error(f"Label file missing for {img_file.name}")
                    return False
                    
                # Validate label format
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    # Remove any trailing % character
                    content = content.rstrip('%')
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) != 5:
                                logger.error(f"Invalid label format in {label_file}: expected 5 parts, got {len(parts)} in line '{line}'")
                                return False
                            try:
                                class_id = int(parts[0])
                                coords = [float(x) for x in parts[1:]]
                                if class_id >= len(self.classes):
                                    logger.error(f"Invalid class ID {class_id} in {label_file}")
                                    return False
                                if not all(0 <= coord <= 1 for coord in coords):
                                    logger.error(f"Invalid coordinates in {label_file}: {coords}")
                                    return False
                            except ValueError as e:
                                logger.error(f"Invalid number format in {label_file}: {e}")
                                return False
                                
            logger.info("Dataset validation passed!")
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False 