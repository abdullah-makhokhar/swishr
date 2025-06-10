#!/usr/bin/env python3
"""
Dataset Preparation Script for Basketball Detection
Prepares the Roboflow basketball dataset for YOLO training
"""

import logging
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from computer_vision.models.dataset_manager import DatasetManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Prepare basketball detection dataset"""
    
    # Configuration
    source_data_dir = "data/basketballDetection.v21i.yolov8"
    output_data_dir = "data/basketball_model"
    
    print("🏀 Basketball Dataset Preparation")
    print("=" * 50)
    
    # Check if source dataset exists
    if not Path(source_data_dir).exists():
        print(f"❌ Source dataset not found: {source_data_dir}")
        return False
        
    # Initialize dataset manager
    print(f"📁 Source: {source_data_dir}")
    print(f"📁 Output: {output_data_dir}")
    
    dataset_manager = DatasetManager(
        data_dir=source_data_dir,
        output_dir=output_data_dir
    )
    
    try:
        # Prepare dataset with proper splits
        print("\n🔄 Preparing dataset splits...")
        dataset_stats = dataset_manager.prepare_dataset(
            train_split=0.7,
            val_split=0.2, 
            test_split=0.1,
            seed=42
        )
        
        print("\n📊 Dataset Statistics:")
        for split_name, stats in dataset_stats.items():
            print(f"\n{split_name.upper()}:")
            print(f"  • Images: {stats['num_images']}")
            print(f"  • Objects: {stats['total_objects']}")
            print(f"  • Avg objects/image: {stats['avg_objects_per_image']:.2f}")
            
            if 'class_distribution' in stats:
                print(f"  • Class distribution:")
                total_objects = stats['total_objects']
                for class_id, count in stats['class_distribution'].items():
                    class_name = ['basketball', 'hoop', 'person'][class_id]
                    percentage = (count / total_objects) * 100 if total_objects > 0 else 0
                    print(f"    - {class_name}: {count} ({percentage:.1f}%)")
                    
        # Validate dataset
        print("\n✅ Validating dataset...")
        if dataset_manager.validate_dataset():
            print("✅ Dataset validation passed!")
        else:
            print("❌ Dataset validation failed!")
            return False
            
        print(f"\n🎯 Dataset prepared successfully!")
        print(f"📝 Report saved: {output_data_dir}/dataset_report.md")
        print(f"⚙️ Config saved: {output_data_dir}/data.yaml")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset preparation failed: {e}")
        logging.error(f"Dataset preparation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 