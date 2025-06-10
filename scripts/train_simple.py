#!/usr/bin/env python3
"""
Simple Basketball Model Training Script
Simplified version for initial testing and validation
"""

import sys
import os
from pathlib import Path
import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    """Simple training with direct YOLO usage"""
    
    print("🏀 Simple Basketball Model Training")
    print("=" * 50)
    
    # Check if dataset exists
    data_config_path = "data/basketball_model/data.yaml"
    if not Path(data_config_path).exists():
        print(f"❌ Dataset not found: {data_config_path}")
        print("Run scripts/prepare_dataset.py first!")
        return False
        
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
        device = "cuda"
    else:
        print("⚠️  CUDA not available, using CPU")
        device = "cpu"
        
    try:
        from ultralytics import YOLO
        
        # Load a pre-trained model
        print("\\n📥 Loading YOLO11n model...")
        model = YOLO('yolo11n.pt')
        
        # Train the model
        print("\\n🚀 Starting training...")
        results = model.train(
            data=data_config_path,
            epochs=10,  # Very short training for testing
            batch=8,    # Small batch size
            imgsz=640,
            device=device,
            project='models/basketball',
            name='simple_v1',
            exist_ok=True,
            verbose=True,
            save=True,
            save_period=5
        )
        
        print("\\n✅ Training completed!")
        
        # Check if model was saved
        model_path = Path("models/basketball/simple_v1/weights/best.pt")
        if model_path.exists():
            print(f"💾 Model saved: {model_path}")
            
            # Quick validation
            print("\\n🔍 Quick validation...")
            val_results = model.val(data=data_config_path)
            
            print(f"📊 Validation Results:")
            print(f"  • mAP50: {getattr(val_results, 'map50', 0):.3f}")
            print(f"  • mAP50-95: {getattr(val_results, 'map', 0):.3f}")
            
            return True
        else:
            print("❌ Model file not found after training")
            return False
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 