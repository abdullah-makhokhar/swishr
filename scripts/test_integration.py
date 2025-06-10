#!/usr/bin/env python3
"""
Integration Test for Enhanced Basketball Detection
Tests integration with existing shot detection pipeline
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from computer_vision.ball_detection_v2 import EnhancedBallDetector
from computer_vision.shot_detection import ShotDetector
from computer_vision.trajectory_analysis import TrajectoryAnalyzer

def test_integration():
    """Test integration with existing shot detection system"""
    
    print("🏀 Basketball Detection Integration Test")
    print("=" * 50)
    
    # Initialize enhanced ball detector
    print("\\n🔧 Initializing Enhanced Ball Detector...")
    ball_detector = EnhancedBallDetector(
        model_path=None,  # Will use fallback for now
        conf_threshold=0.3,
        fallback_to_color=True,
        enable_tracking=True
    )
    
    # Initialize existing components
    print("🎯 Initializing Shot Detector...")
    shot_detector = ShotDetector()
    
    print("📈 Initializing Trajectory Analyzer...")
    trajectory_analyzer = TrajectoryAnalyzer()
    
    # Create test sequence simulating a basketball shot
    print("\\n🎬 Creating test shot sequence...")
    test_frames = create_shot_sequence()
    
    print(f"📹 Generated {len(test_frames)} test frames")
    
    # Process frames through the pipeline
    print("\\n🔄 Processing frames through detection pipeline...")
    
    all_detections = []
    shot_events = []
    trajectories = []
    
    for i, frame in enumerate(test_frames):
        print(f"\\rProcessing frame {i+1}/{len(test_frames)}", end="", flush=True)
        
        # Enhanced ball detection
        detections = ball_detector.detect_balls(frame, use_tracking=True)
        all_detections.append(detections)
        
        # Convert to legacy format for existing shot detector
        legacy_balls = []
        for detection in detections:
            center = detection['center']
            radius = detection['radius']
            legacy_balls.append((center[0], center[1], radius))
            
        # Shot detection
        current_shot, annotated_frame = shot_detector.process_frame(frame)
        if current_shot:
            shot_events.append({
                'frame': i,
                'shot_id': current_shot.shot_id,
                'phase': current_shot.phase.value,
                'balls': legacy_balls
            })
            
        # Trajectory analysis
        if legacy_balls:
            # Create BallPosition object for trajectory analyzer
            from computer_vision.ball_detection import BallPosition
            ball_center = legacy_balls[0][:2]  # Use first detected ball
            ball_pos = BallPosition(
                x=ball_center[0],
                y=ball_center[1],
                timestamp=time.time(),
                confidence=0.8
            )
            trajectory_analyzer.add_ball_position(ball_pos)
            
    print("\\n")
    
    # Analyze results
    print("\\n📊 Analysis Results:")
    
    # Detection statistics
    total_detections = sum(len(detections) for detections in all_detections)
    frames_with_detections = sum(1 for detections in all_detections if detections)
    
    print(f"\\n🎯 Detection Statistics:")
    print(f"  • Total detections: {total_detections}")
    print(f"  • Frames with detections: {frames_with_detections}/{len(test_frames)}")
    print(f"  • Detection rate: {frames_with_detections/len(test_frames)*100:.1f}%")
    
    # Shot detection results
    print(f"\\n🏀 Shot Detection Events:")
    for event in shot_events:
        print(f"  • Frame {event['frame']}: {event['phase']} (ID: {event['shot_id']}, {len(event['balls'])} balls)")
        
    # Trajectory analysis
    current_trajectory = trajectory_analyzer.analyze_current_trajectory()
    if current_trajectory:
        print(f"\\n📈 Trajectory Analysis:")
        print(f"  • Trajectory points: {len(current_trajectory.points)}")
        print(f"  • Arc angle: {current_trajectory.arc_angle or 0:.1f}°")
        print(f"  • Entry angle: {current_trajectory.entry_angle or 0:.1f}°")
        print(f"  • Peak height: {current_trajectory.peak_height or 0:.1f}px")
        print(f"  • Shot distance: {current_trajectory.shot_distance or 0:.1f}px")
        print(f"  • Confidence: {current_trajectory.confidence:.3f}")
    else:
        print(f"\\n📈 Trajectory Analysis:")
        print(f"  • Insufficient trajectory data for analysis")
                
    # Performance statistics
    performance_stats = ball_detector.get_performance_stats()
    print(f"\\n⚡ Performance Statistics:")
    print(f"  • Avg detection time: {performance_stats['avg_detection_time']:.3f}s")
    print(f"  • Estimated FPS: {1/performance_stats['avg_detection_time']:.1f}")
    print(f"  • Success rate: {performance_stats['success_rate']:.3f}")
    print(f"  • Model available: {performance_stats['model_available']}")
    
    # Test with trained model if available
    test_trained_model_integration()
    
    return True

def create_shot_sequence(num_frames=30):
    """Create a sequence of frames simulating a basketball shot"""
    
    frames = []
    width, height = 640, 480
    
    # Basketball trajectory (parabolic arc)
    start_x, start_y = 100, 400  # Bottom left
    end_x, end_y = 540, 350      # Top right (hoop area)
    peak_y = 200                 # Arc peak
    
    for i in range(num_frames):
        # Create frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add court background (green)
        frame[:, :] = (34, 139, 34)  # Forest green
        
        # Calculate basketball position along parabolic trajectory
        t = i / (num_frames - 1)  # Normalized time [0, 1]
        
        # Parabolic trajectory
        x = start_x + (end_x - start_x) * t
        y_linear = start_y + (end_y - start_y) * t
        y_arc = y_linear - 4 * (peak_y - max(start_y, end_y)) * t * (1 - t)
        
        # Draw basketball
        center = (int(x), int(y_arc))
        radius = 25
        
        # Orange basketball
        cv2.circle(frame, center, radius, (0, 165, 255), -1)
        
        # Basketball lines
        cv2.line(frame, (center[0] - radius, center[1]), 
                (center[0] + radius, center[1]), (0, 0, 0), 2)
        cv2.line(frame, (center[0], center[1] - radius), 
                (center[0], center[1] + radius), (0, 0, 0), 2)
        
        # Add some noise for realism
        noise = np.random.randint(0, 10, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Draw hoop at the end
        hoop_center = (end_x, end_y)
        cv2.circle(frame, hoop_center, 30, (0, 0, 255), 3)  # Red hoop
        cv2.rectangle(frame, (hoop_center[0] - 40, hoop_center[1] - 5),
                     (hoop_center[0] + 40, hoop_center[1] + 5), (139, 69, 19), -1)  # Brown backboard
        
        frames.append(frame)
        
    return frames

def test_trained_model_integration():
    """Test integration with trained model if available"""
    
    # Check for trained models
    model_paths = [
        "models/basketball/simple_v1/weights/best.pt",
        "models/basketball/basketball_v1/weights/best.pt"
    ]
    
    trained_model_path = None
    for path in model_paths:
        if Path(path).exists():
            trained_model_path = path
            break
            
    if trained_model_path:
        print(f"\\n🎯 Testing with trained model: {trained_model_path}")
        
        try:
            # Initialize detector with trained model
            trained_detector = EnhancedBallDetector(
                model_path=trained_model_path,
                conf_threshold=0.25,
                fallback_to_color=True,
                enable_tracking=True
            )
            
            # Test with a single frame
            test_frame = create_shot_sequence(1)[0]
            detections = trained_detector.detect_balls(test_frame)
            
            print(f"✅ Trained model integration test:")
            print(f"  • Detections: {len(detections)}")
            print(f"  • Model available: {trained_detector.model_available}")
            
            if detections:
                detection = detections[0]
                print(f"  • Best detection confidence: {detection['confidence']:.3f}")
                print(f"  • Detection method: {detection['detection_method']}")
                
            # Performance comparison
            stats = trained_detector.get_performance_stats()
            if 'model_stats' in stats:
                model_stats = stats['model_stats']
                print(f"  • Model avg confidence: {model_stats.get('avg_confidence', 0):.3f}")
                
        except Exception as e:
            print(f"❌ Trained model integration failed: {e}")
            
    else:
        print("\\n⏳ No trained model available for integration test")

def save_test_video():
    """Save test sequence as video for visual inspection"""
    
    print("\\n🎥 Saving test video...")
    
    frames = create_shot_sequence(60)  # Longer sequence
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_shot_sequence.mp4', fourcc, 10.0, (640, 480))
    
    for frame in frames:
        out.write(frame)
        
    out.release()
    print("✅ Test video saved: test_shot_sequence.mp4")

if __name__ == "__main__":
    try:
        # Run integration test
        success = test_integration()
        
        # Save test video for visual inspection
        save_test_video()
        
        print("\\n🎉 Integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 