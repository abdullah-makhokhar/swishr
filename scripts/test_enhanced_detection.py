#!/usr/bin/env python3
"""
Test Enhanced Ball Detection System
Tests the new basketball-specific detection with fallback to color detection
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from computer_vision.ball_detection_v2 import EnhancedBallDetector, detect_basketball

def test_enhanced_detector():
    """Test the enhanced ball detector"""
    
    print("🏀 Testing Enhanced Ball Detection System")
    print("=" * 50)
    
    # Test with fallback (no trained model yet)
    print("\\n🔧 Testing with color fallback (no trained model)...")
    detector_fallback = EnhancedBallDetector(
        model_path=None,  # No model path - will use fallback
        conf_threshold=0.3,
        fallback_to_color=True,
        enable_tracking=False
    )
    
    # Create a test image with orange circle (simulated basketball)
    test_image = create_test_basketball_image()
    
    # Test detection
    start_time = time.time()
    detections = detector_fallback.detect_balls(test_image)
    detection_time = time.time() - start_time
    
    print(f"✅ Fallback detection completed in {detection_time:.3f}s")
    print(f"📊 Found {len(detections)} basketball(s)")
    
    for i, detection in enumerate(detections):
        print(f"  Ball {i+1}:")
        print(f"    • Center: {detection['center']}")
        print(f"    • Radius: {detection['radius']}")
        print(f"    • Confidence: {detection['confidence']:.3f}")
        print(f"    • Method: {detection['detection_method']}")
        
    # Test performance stats
    stats = detector_fallback.get_performance_stats()
    print(f"\\n📈 Performance Stats:")
    print(f"  • Model available: {stats['model_available']}")
    print(f"  • Fallback enabled: {stats['fallback_enabled']}")
    print(f"  • Success rate: {stats['success_rate']:.3f}")
    print(f"  • Avg detection time: {stats['avg_detection_time']:.3f}s")
    
    # Test compatibility function
    print("\\n🔄 Testing compatibility function...")
    legacy_detections = detect_basketball(test_image)
    print(f"✅ Legacy format: {len(legacy_detections)} detections")
    for i, (x, y, r) in enumerate(legacy_detections):
        print(f"  Ball {i+1}: center=({x}, {y}), radius={r}")
        
    # Test with multiple basketballs
    print("\\n🏀 Testing with multiple basketballs...")
    multi_ball_image = create_multi_basketball_image()
    multi_detections = detector_fallback.detect_balls(multi_ball_image)
    print(f"✅ Multi-ball detection: {len(multi_detections)} basketballs found")
    
    # Test performance with multiple frames
    print("\\n⚡ Performance test (100 frames)...")
    performance_times = []
    for i in range(100):
        # Create slightly different test images
        test_img = create_test_basketball_image(noise_level=i % 10)
        start = time.time()
        detections = detector_fallback.detect_balls(test_img)
        performance_times.append(time.time() - start)
        
    avg_time = np.mean(performance_times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"✅ Average detection time: {avg_time:.3f}s")
    print(f"🚀 Estimated FPS: {fps:.1f}")
    
    # Final stats
    final_stats = detector_fallback.get_performance_stats()
    print(f"\\n📊 Final Performance Stats:")
    print(f"  • Total detections: {final_stats['total_detections']}")
    print(f"  • Successful detections: {final_stats['successful_detections']}")
    print(f"  • Success rate: {final_stats['success_rate']:.3f}")
    
    return True

def create_test_basketball_image(width=640, height=480, noise_level=0):
    """Create a test image with a basketball-like object"""
    
    # Create blank image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some background noise if specified
    if noise_level > 0:
        noise = np.random.randint(0, noise_level, (height, width, 3), dtype=np.uint8)
        image = cv2.add(image, noise)
    
    # Draw basketball (orange circle with black lines)
    center = (width // 2, height // 2)
    radius = 40
    
    # Orange basketball
    cv2.circle(image, center, radius, (0, 165, 255), -1)  # Orange in BGR
    
    # Basketball lines
    cv2.line(image, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (0, 0, 0), 2)
    cv2.line(image, (center[0], center[1] - radius), (center[0], center[1] + radius), (0, 0, 0), 2)
    
    # Add some curved lines for realism
    cv2.ellipse(image, center, (radius, radius//2), 0, 0, 180, (0, 0, 0), 2)
    cv2.ellipse(image, center, (radius//2, radius), 90, 0, 180, (0, 0, 0), 2)
    
    return image

def create_multi_basketball_image(width=640, height=480):
    """Create test image with multiple basketballs"""
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Basketball positions and sizes
    basketballs = [
        ((150, 150), 30),  # Top left
        ((500, 150), 35),  # Top right  
        ((320, 350), 40),  # Bottom center
    ]
    
    for (center, radius) in basketballs:
        # Orange basketball
        cv2.circle(image, center, radius, (0, 165, 255), -1)
        
        # Basketball lines
        cv2.line(image, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (0, 0, 0), 2)
        cv2.line(image, (center[0], center[1] - radius), (center[0], center[1] + radius), (0, 0, 0), 2)
        
    return image

def test_with_trained_model():
    """Test with trained model if available"""
    
    # Check if trained model exists
    model_paths = [
        "models/basketball/basketball_v1/weights/best.pt",
        "models/basketball/basketball_v1/weights/last.pt"
    ]
    
    trained_model_path = None
    for path in model_paths:
        if Path(path).exists():
            trained_model_path = path
            break
            
    if trained_model_path:
        print(f"\\n🎯 Testing with trained model: {trained_model_path}")
        
        try:
            detector_trained = EnhancedBallDetector(
                model_path=trained_model_path,
                conf_threshold=0.3,
                fallback_to_color=True,
                enable_tracking=True
            )
            
            test_image = create_test_basketball_image()
            detections = detector_trained.detect_balls(test_image)
            
            print(f"✅ Trained model detection: {len(detections)} basketballs")
            
            stats = detector_trained.get_performance_stats()
            print(f"📊 Model Stats:")
            print(f"  • Model available: {stats['model_available']}")
            print(f"  • Tracking enabled: {stats['tracking_enabled']}")
            
            if 'model_stats' in stats:
                model_stats = stats['model_stats']
                print(f"  • Active tracks: {model_stats.get('active_tracks', 0)}")
                print(f"  • Avg confidence: {model_stats.get('avg_confidence', 0):.3f}")
                
        except Exception as e:
            print(f"❌ Trained model test failed: {e}")
            
    else:
        print("\\n⏳ No trained model found yet. Training may still be in progress.")

if __name__ == "__main__":
    try:
        # Test enhanced detector
        success = test_enhanced_detector()
        
        # Test with trained model if available
        test_with_trained_model()
        
        print("\\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1) 