# 🏀 Basketball Analysis System - Quick Start Guide

## Complete Setup & Usage Instructions

Follow these steps in exact order to set up and run the basketball analysis system.

---

## 📋 Prerequisites Check

Run these commands to verify your system is ready:

```bash
# Check Python version (3.9+ required)
python --version

# Check Git installation
git --version

# Check available memory (8GB+ recommended)
# macOS/Linux:
free -h
# Windows:
systeminfo | findstr "Total Physical Memory"
```

---

## 🚀 Step 1: Installation

### 1.1 Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/abdullah-makhokhar/swishr.git
cd swishr

# Create virtual environment
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2, ultralytics; print('✅ Installation successful')"
```

### 1.2 Set Environment Variables
```bash
# Set Python path for development
export PYTHONPATH=$PWD/src:$PYTHONPATH

# Optional: Set for permanent use
echo "export PYTHONPATH=$PWD/src:\$PYTHONPATH" >> ~/.bashrc
```

---

## 📊 Step 2: Dataset Preparation

### 2.1 Validate Dataset
```bash
# Prepare and validate the basketball dataset
python scripts/prepare_dataset.py
```

**Expected Output:**
```
🏀 Basketball Dataset Preparation
==================================================
✅ Dataset found: 2,955 images
✅ Labels verified: basketball(0), hoop(1), person(2)
📊 Train: 2,364 images | Val: 591 images
✅ Dataset ready for training
```

### 2.2 Dataset Analysis (Optional)
```bash
# Get detailed dataset statistics
python -c "
from src.computer_vision.models import DatasetManager
dm = DatasetManager('data/basketballDetection.v21i.yolov8')
dm.analyze_dataset()
"
```

---

## 🎯 Step 3: Model Training

### Option A: Quick Training (Recommended for First Run)
```bash
# Fast 10-epoch training for testing
python scripts/train_simple.py
```

**What to expect:**
- ⏱️ Duration: 15-30 minutes
- 📈 10 epochs of training
- 💾 Model saved to: `models/basketball/simple_v1/`

### Option B: Full Production Training
```bash
# Comprehensive 100-epoch training
python scripts/train_basketball_model.py
```

**What to expect:**
- ⏱️ Duration: 2-4 hours
- 📈 100 epochs with optimization
- 💾 Model saved to: `models/basketball/basketball_v1/`

### Monitor Training Progress
```bash
# Watch training in real-time (in another terminal)
tail -f models/basketball/*/train/results.csv

# Or check training status
ls -la models/basketball/
```

---

## ✅ Step 4: Testing & Validation

### 4.1 Test Enhanced Detection
```bash
# Test the basketball-specific detection
python scripts/test_enhanced_detection.py
```

### 4.2 Integration Test
```bash
# Test the complete pipeline
python scripts/test_integration.py
```

### 4.3 Final Validation
```bash
# Comprehensive system validation
python scripts/final_validation.py
```

**Expected Results:**
```
🏀 Basketball Analysis System Validation
==========================================
✅ Dataset: OK (2,955 images)
✅ Model: OK (basketball_v1)
✅ Detection: OK (95.2% accuracy)
✅ Tracking: OK (ByteTrack initialized)
✅ Performance: OK (30.5 FPS)
==========================================
🎉 System ready for basketball analysis!
```

---

## 🎬 Step 5: Running Enhanced Analysis

### 5.1 List Available Models
```bash
# First, check what models you have trained
python run_basketball_analysis.py --list-models
```

**Expected Output:**
```
📋 AVAILABLE BASKETBALL MODELS
----------------------------------------
  🎯 simple_v1
     Path: models/basketball/simple_v1/weights/best.pt
     Size: 5.2 MB
     Classes: 3
```

### 5.2 Basic Analysis
```bash
# Real-time camera with auto-model selection
python run_basketball_analysis.py --camera 0

# Analyze video with specific model
python run_basketball_analysis.py --input your_video.mp4 --model simple_v1

# Quick analysis with JSON results
python run_basketball_analysis.py --input video.mp4 --save-json
```

### 5.3 High-Performance Analysis
```bash
# Enhanced analysis with all features
python run_basketball_analysis.py \
    --input your_video.mp4 \
    --model simple_v1 \
    --confidence 0.8 \
    --tracker bytetrack \
    --output enhanced_output.mp4 \
    --save-json \
    --benchmark

# Batch processing (fastest, no display)
python run_basketball_analysis.py \
    --input video.mp4 \
    --batch \
    --no-display \
    --output-dir results/ \
    --model simple_v1
```

### 5.4 Camera Analysis Options
```bash
# HD camera analysis
python run_basketball_analysis.py \
    --camera 0 \
    --width 1280 \
    --height 720 \
    --model simple_v1 \
    --confidence 0.75

# Multiple camera support
python run_basketball_analysis.py --camera 1  # External camera
python run_basketball_analysis.py --camera 2  # USB camera
```

---

## 📈 Step 6: Performance Monitoring

### 6.1 Check System Performance
```bash
# Benchmark the system
python -c "
from src.computer_vision.ball_detection_v2 import EnhancedBallDetector
detector = EnhancedBallDetector()
print('✅ Enhanced detector loaded successfully')
print(f'📊 Model: {detector.model_path}')
print(f'🎯 Classes: {detector.model.names}')
"
```

### 6.2 Memory and Speed Check
```bash
# Quick performance test
python -c "
import time
import psutil
from src.computer_vision.ball_detection_v2 import EnhancedBallDetector

print('🔄 Loading detector...')
start_time = time.time()
detector = EnhancedBallDetector()
load_time = time.time() - start_time

print(f'⚡ Load time: {load_time:.2f}s')
print(f'💾 Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB')
print('✅ Performance check complete')
"
```

---

## 🔧 Command Reference

### Essential Commands
```bash
# Full system setup (run once)
git clone https://github.com/abdullah-makhokhar/swishr.git && cd swishr
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && python scripts/prepare_dataset.py

# Quick training and validation (run after setup)
python scripts/train_simple.py && python scripts/test_integration.py

# List available models and run analysis
python run_basketball_analysis.py --list-models
python run_basketball_analysis.py --input video.mp4 --model simple_v1

# Run real-time analysis with enhanced detection
python run_basketball_analysis.py --camera 0 --model auto
```

### Development Commands
```bash
# Dataset management
python scripts/prepare_dataset.py                    # Prepare dataset
python scripts/prepare_dataset.py --validate         # Validate only

# Training options
python scripts/train_simple.py                       # Quick training (10 epochs)
python scripts/train_basketball_model.py             # Full training (100 epochs)
python scripts/train_basketball_model.py --epochs 50 # Custom epochs

# Testing suite
python scripts/test_enhanced_detection.py            # Test detection
python scripts/test_integration.py                   # Test integration
python scripts/final_validation.py                   # Full validation
pytest tests/ -v --cov=src                          # Unit tests

# Enhanced analysis options
python run_basketball_analysis.py --list-models                    # List available models
python run_basketball_analysis.py --camera 0 --model simple_v1     # Camera with specific model
python run_basketball_analysis.py --input video.mp4 --save-json    # Video with JSON output
python run_basketball_analysis.py --input video.mp4 --batch        # Batch mode (fastest)
python run_basketball_analysis.py --help                           # Show all options
```

---

## 🚨 Troubleshooting

### Common Issues & Solutions

**1. Import Error: Module not found**
```bash
# Solution: Set Python path
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

**2. Training fails with YOLO argument errors**
```bash
# Solution: Update ultralytics
pip install ultralytics --upgrade
```

**3. Low detection accuracy**
```bash
# Solution: Validate dataset and retrain
python scripts/prepare_dataset.py --validate
python scripts/train_simple.py
```

**4. Performance issues**
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check system resources
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().available/1024/1024/1024:.1f}GB available')"
```

### Getting Help
```bash
# System diagnostics
python scripts/final_validation.py

# Check installation
python -c "
import cv2, ultralytics, torch
print('✅ OpenCV:', cv2.__version__)
print('✅ Ultralytics:', ultralytics.__version__)
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
"
```

---

## 📋 Quick Reference Card

| Action | Command |
|--------|---------|
| **Setup** | `git clone repo && cd swishr && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt` |
| **Prepare Data** | `python scripts/prepare_dataset.py` |
| **Quick Train** | `python scripts/train_simple.py` |
| **Test System** | `python scripts/test_integration.py` |
| **Analyze Video** | `python run_basketball_analysis.py --input video.mp4` |
| **Real-time** | `python run_basketball_analysis.py --camera 0` |
| **Validate** | `python scripts/final_validation.py` |

---

## 🎯 Success Indicators

After following this guide, you should see:

✅ **Installation Success**: All imports work without errors  
✅ **Dataset Ready**: 2,955 images prepared and validated  
✅ **Model Trained**: Model saved to `models/basketball/`  
✅ **Tests Pass**: All validation scripts complete successfully  
✅ **Analysis Working**: Video/camera analysis produces results  
✅ **Performance**: 30+ FPS, <100ms latency achieved  

**Congratulations! Your basketball analysis system is ready! 🏀** 