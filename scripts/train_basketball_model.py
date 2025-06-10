#!/usr/bin/env python3
"""
Basketball Model Training Script
Trains a basketball-specific YOLO model using the prepared dataset
"""

import logging
import sys
import os
from pathlib import Path
import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from computer_vision.models.model_trainer import ModelTrainer
from computer_vision.models.basketball_yolo import BasketballYOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Train basketball detection model"""
    
    # Configuration
    data_config_path = "data/basketball_model/data.yaml"
    output_dir = "models/basketball"
    
    print("🏀 Basketball Model Training")
    print("=" * 50)
    
    # Check if dataset exists
    if not Path(data_config_path).exists():
        print(f"❌ Dataset not found: {data_config_path}")
        print("Run scripts/prepare_dataset.py first!")
        return False
        
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️  CUDA not available, using CPU (training will be slower)")
        
    # Initialize trainer
    print(f"📁 Data config: {data_config_path}")
    print(f"📁 Output: {output_dir}")
    
    trainer = ModelTrainer(
        data_config_path=data_config_path,
        base_model="yolo11n.pt",  # Start with nano model for faster training
        project_name="basketball_detection",
        experiment_name="basketball_v1",
        output_dir=output_dir
    )
    
    try:
        # Prepare optimized training configuration
        print("\\n⚙️ Preparing training configuration...")
        
        training_config = trainer.prepare_training_config(
            epochs=50,              # Reduced epochs for initial training
            batch_size=16,          # Adjust based on GPU memory
            imgsz=640,              # Standard YOLO image size
            lr0=0.01,               # Initial learning rate
            lrf=0.1,                # Final learning rate factor
            warmup_epochs=3,        # Warmup epochs
            augment=True,           # Enable data augmentation
            
            # Basketball-specific optimizations
            cls_gain=0.5,           # Class loss gain
            box_gain=0.05,          # Box loss gain
            obj_gain=1.0,           # Object loss gain
            
            # Validation settings
            save_period=10,         # Save checkpoint every 10 epochs
            workers=4,              # Number of workers (adjust for your system)
            cache=False,            # Don't cache images (saves memory)
            verbose=True
        )
        
        print("\\n📋 Training Configuration:")
        key_params = ['epochs', 'batch', 'imgsz', 'lr0', 'device']
        for param in key_params:
            if param in training_config:
                print(f"  • {param}: {training_config[param]}")
                
        # Start training
        print("\\n🚀 Starting training...")
        training_results = trainer.train_model(
            training_config=training_config,
            use_wandb=False,  # Disable W&B for now
            resume=False
        )
        
        print("\\n✅ Training completed!")
        print(f"⏱️  Training time: {training_results['training_time']:.2f} seconds")
        print(f"💾 Best model: {training_results['best_model_path']}")
        
        # Validate the trained model
        print("\\n🔍 Validating trained model...")
        validation_results = trainer.validate_model(
            model_path=training_results['best_model_path'],
            conf_threshold=0.001,
            iou_threshold=0.6
        )
        
        print("\\n📊 Validation Results:")
        metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
        for metric in metrics:
            if metric in validation_results:
                print(f"  • {metric}: {validation_results[metric]:.3f}")
                
        # Test on test set
        print("\\n🧪 Testing on test set...")
        test_results = trainer.test_model(
            model_path=training_results['best_model_path'],
            test_images_dir="data/basketball_model/test/images",
            conf_threshold=0.25,
            iou_threshold=0.45
        )
        
        print("\\n📈 Test Results:")
        print(f"  • Total detections: {test_results['total_detections']}")
        print(f"  • Avg detections/image: {test_results['avg_detections_per_image']:.2f}")
        print(f"  • Avg confidence: {test_results['avg_confidence']:.3f}")
        
        class_detections = test_results['class_detections']
        print("  • Class detections:")
        for class_name, count in class_detections.items():
            print(f"    - {class_name}: {count}")
            
        # Test the custom basketball model
        print("\\n🏀 Testing BasketballYOLO integration...")
        basketball_model = BasketballYOLO(
            model_path=training_results['best_model_path'],
            conf_threshold=0.3,
            iou_threshold=0.7
        )
        
        # Get performance stats
        performance_stats = basketball_model.get_performance_stats()
        print(f"✅ BasketballYOLO model loaded successfully!")
        
        print(f"\\n🎯 Training Summary:")
        print(f"  • Model: {training_results['best_model_path']}")
        print(f"  • mAP50: {validation_results.get('mAP50', 0):.3f}")
        print(f"  • mAP50-95: {validation_results.get('mAP50-95', 0):.3f}")
        print(f"  • Test images: {test_results['total_test_images']}")
        print(f"  • Test detections: {test_results['total_detections']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logging.error(f"Training failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 