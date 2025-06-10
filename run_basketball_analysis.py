#!/usr/bin/env python3
"""
Enhanced Basketball Shot Analysis Runner
State-of-the-art basketball analysis with custom-trained models and advanced tracking
"""

import argparse
import json
import sys
import os
import cv2
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from computer_vision.ball_detection_v2 import EnhancedBallDetector
from computer_vision.shot_detection import ShotDetector
from computer_vision.pose_estimation import PoseEstimator
from computer_vision.trajectory_analysis import TrajectoryAnalyzer


def print_banner():
    """Print enhanced application banner"""
    print("=" * 70)
    print("🏀 ENHANCED BASKETBALL SHOT ANALYSIS - swishr AI v2.0")
    print("=" * 70)
    print("🚀 Custom YOLO Models • ByteTrack • Professional Analytics")
    print()


def print_system_info():
    """Print system information and capabilities"""
    print("📊 SYSTEM CAPABILITIES")
    print("-" * 30)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"🔧 PyTorch: {torch.__version__}")
        print(f"🎮 CUDA Available: {'✅ Yes' if cuda_available else '❌ No (CPU only)'}")
        
        from ultralytics import __version__
        print(f"🎯 Ultralytics: {__version__}")
        
        print(f"📹 OpenCV: {cv2.__version__}")
        
    except ImportError as e:
        print(f"⚠️ System check failed: {e}")
    
    print()


def list_available_models() -> Dict[str, str]:
    """List all available trained basketball models"""
    models = {}
    models_dir = Path("models/basketball")
    
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                # Look for best.pt and last.pt files
                best_model = model_dir / "weights" / "best.pt"
                last_model = model_dir / "weights" / "last.pt"
                
                if best_model.exists():
                    models[model_dir.name] = str(best_model)
                elif last_model.exists():
                    models[model_dir.name] = str(last_model)
                    
    return models


def get_model_info(model_path: str) -> Dict[str, Any]:
    """Get information about a trained model"""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        info = {
            "classes": model.names if hasattr(model, 'names') else {},
            "num_classes": len(model.names) if hasattr(model, 'names') else 0,
            "model_size": os.path.getsize(model_path) / (1024 * 1024),  # MB
        }
        
        return info
    except Exception as e:
        return {"error": str(e)}


def setup_video_source(source: str, width: int = 640, height: int = 480) -> Optional[cv2.VideoCapture]:
    """Setup video input from camera or file with enhanced error handling"""
    try:
        # Try camera input first
        if source.isdigit():
            camera_idx = int(source)
            cap = cv2.VideoCapture(camera_idx)
            
            if cap.isOpened():
                # Configure camera
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
                
                print(f"📹 Camera {camera_idx} initialized ({width}x{height} @ 30fps)")
                return cap
            else:
                print(f"❌ Could not open camera {camera_idx}")
                return None
                
        # Try file input
        if not os.path.exists(source):
            print(f"❌ Video file not found: {source}")
            return None
            
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Could not open video file: {source}")
            return None
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📹 Video loaded: {source}")
        print(f"   📊 {total_frames} frames @ {fps:.1f}fps ({duration:.1f}s)")
        
        return cap
        
    except Exception as e:
        print(f"❌ Error setting up video source: {e}")
        return None


def setup_video_writer(output_path: str, width: int, height: int, fps: float) -> Optional[cv2.VideoWriter]:
    """Setup video output writer"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Try different codecs for compatibility
        codecs = ['mp4v', 'XVID', 'MJPG']
        
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if writer.isOpened():
                print(f"💾 Video output: {output_path} (codec: {codec})")
                return writer
                
        print(f"❌ Could not create video writer for: {output_path}")
        return None
        
    except Exception as e:
        print(f"❌ Error setting up video writer: {e}")
        return None


def create_enhanced_overlay(frame: cv2.Mat, detector: EnhancedBallDetector, 
                          shot_detector: ShotDetector, fps: float) -> cv2.Mat:
    """Create enhanced information overlay"""
    overlay = frame.copy()
    height, width = frame.shape[:2]
    
    # Semi-transparent background
    cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Title
    cv2.putText(frame, "🏀 BASKETBALL ANALYSIS", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Model info
    model_name = Path(detector.model_path).parent.name if detector.model_path else "default"
    cv2.putText(frame, f"Model: {model_name}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Performance metrics
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Detection stats
    stats = shot_detector.get_shooting_statistics() if shot_detector else {}
    
    y_pos = 100
    metrics = [
        ("Total Shots", stats.get('total_shots', 0)),
        ("Shooting %", f"{stats.get('shooting_percentage', 0):.1f}%"),
        ("Avg Confidence", f"{stats.get('avg_confidence', 0)*100:.1f}%"),
        ("Arc Quality", f"{stats.get('avg_arc_consistency', 0):.1f}%"),
    ]
    
    for label, value in metrics:
        cv2.putText(frame, f"{label}: {value}", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += 20
    
    # Controls info
    cv2.putText(frame, "Press 'q' to quit, 'r' to reset", (20, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return frame


def save_analysis_results(shot_detector: ShotDetector, output_file: str, 
                         model_info: Dict[str, Any], processing_time: float) -> bool:
    """Save comprehensive analysis results"""
    try:
        results = {
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "model_info": model_info,
            "session_stats": shot_detector.get_shooting_statistics(),
            "shots": shot_detector.get_shot_data() if hasattr(shot_detector, 'get_shot_data') else [],
            "system_info": {
                "version": "swishr v2.0",
                "enhanced_detection": True,
                "custom_basketball_model": True,
                "bytetrack_enabled": True
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        return True
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False


def main():
    """Enhanced main function with comprehensive options"""
    parser = argparse.ArgumentParser(
        description='Enhanced Basketball Shot Analysis with Custom Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time camera analysis
  python run_basketball_analysis.py --camera 0
  
  # Analyze video with custom model
  python run_basketball_analysis.py --input video.mp4 --model basketball_v1
  
  # High-performance batch processing
  python run_basketball_analysis.py --input video.mp4 --batch --no-display --output results/
  
  # List available models
  python run_basketball_analysis.py --list-models
        """
    )
    
    # Input/Output options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', help='Input video file path')
    input_group.add_argument('--camera', '-c', type=int, help='Camera device index (0, 1, 2...)')
    input_group.add_argument('--list-models', action='store_true', help='List available trained models')
    
    parser.add_argument('--output', '-o', help='Output video file path')
    parser.add_argument('--output-dir', default='results', help='Output directory for results (default: results)')
    parser.add_argument('--save-json', action='store_true', help='Save analysis results to JSON')
    
    # Model options
    parser.add_argument('--model', default='auto', 
                       help='Model to use: auto, model_name, or path to .pt file (default: auto)')
    parser.add_argument('--confidence', type=float, default=0.7, 
                       help='Detection confidence threshold (default: 0.7)')
    parser.add_argument('--iou-threshold', type=float, default=0.5, 
                       help='IoU threshold for NMS (default: 0.5)')
    
    # Tracking options
    parser.add_argument('--tracker', choices=['bytetrack', 'botsort', 'none'], default='bytetrack',
                       help='Tracking algorithm to use (default: bytetrack)')
    parser.add_argument('--track-buffer', type=int, default=30, 
                       help='Tracking buffer size (default: 30)')
    
    # Processing options
    parser.add_argument('--batch', action='store_true', help='Batch processing mode (faster, no real-time display)')
    parser.add_argument('--no-display', action='store_true', help='Disable video display (headless mode)')
    parser.add_argument('--fps-limit', type=int, default=30, help='Limit processing FPS (default: 30)')
    
    # Video options
    parser.add_argument('--width', type=int, default=640, help='Frame width for camera (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Frame height for camera (default: 480)')
    
    # Debug options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--benchmark', action='store_true', help='Show performance benchmarks')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Handle list-models option
    if args.list_models:
        models = list_available_models()
        
        if not models:
            print("❌ No trained models found in models/basketball/")
            print("💡 Run 'python scripts/train_simple.py' to train a model first")
            return 1
            
        print("📋 AVAILABLE BASKETBALL MODELS")
        print("-" * 40)
        for name, path in models.items():
            info = get_model_info(path)
            size_mb = info.get('model_size', 0)
            num_classes = info.get('num_classes', 0)
            print(f"  🎯 {name}")
            print(f"     Path: {path}")
            print(f"     Size: {size_mb:.1f} MB")
            print(f"     Classes: {num_classes}")
            print()
        return 0
    
    print_system_info()
    
    # Setup video source
    if args.camera is not None:
        cap = setup_video_source(str(args.camera), args.width, args.height)
        source_name = f"Camera {args.camera}"
    else:
        cap = setup_video_source(args.input)
        source_name = args.input
        
    if cap is None:
        return 1
        
    # Get frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Setup video writer if needed
    writer = None
    if args.output:
        writer = setup_video_writer(args.output, width, height, source_fps)
    
    # Initialize enhanced detection system
    print("🚀 INITIALIZING ENHANCED DETECTION SYSTEM")
    print("-" * 50)
    
    try:
        # Select model
        if args.model == 'auto':
            models = list_available_models()
            if models:
                model_path = list(models.values())[0]  # Use first available model
                print(f"🎯 Auto-selected model: {Path(model_path).parent.name}")
            else:
                print("⚠️ No custom models found, using pre-trained YOLO")
                model_path = None
        elif args.model.endswith('.pt'):
            model_path = args.model
        else:
            models = list_available_models()
            if args.model in models:
                model_path = models[args.model]
            else:
                print(f"❌ Model '{args.model}' not found")
                print(f"Available models: {list(models.keys())}")
                return 1
        
        # Initialize enhanced ball detector
        detector = EnhancedBallDetector(
            model_path=model_path,
            conf_threshold=args.confidence,
            iou_threshold=args.iou_threshold,
            enable_tracking=args.tracker != 'none'
        )
        
        # Initialize other components
        pose_estimator = PoseEstimator()
        trajectory_analyzer = TrajectoryAnalyzer()
        
        shot_detector = ShotDetector(
            ball_detector=detector,
            pose_estimator=pose_estimator,
            trajectory_analyzer=trajectory_analyzer
        )
        
        print("✅ Enhanced detection system initialized")
        
        # Get model info for results
        model_info = get_model_info(model_path) if model_path else {"type": "pre-trained"}
        
    except Exception as e:
        print(f"❌ Failed to initialize detection system: {e}")
        return 1
    
    # Processing loop
    print(f"\n🎬 PROCESSING: {source_name}")
    print("=" * 50)
    
    frame_count = 0
    start_time = time.time()
    fps_counter = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process frame through enhanced pipeline
            current_shot, annotated_frame = shot_detector.process_frame(frame)
            
            # Calculate FPS
            fps_frame_count += 1
            if time.time() - fps_counter >= 1.0:
                current_fps = fps_frame_count / (time.time() - fps_counter)
                fps_counter = time.time()
                fps_frame_count = 0
            
            # Add enhanced overlay
            if not args.batch:
                annotated_frame = create_enhanced_overlay(
                    annotated_frame, detector, shot_detector, current_fps
                )
            
            # Write to output video
            if writer:
                writer.write(annotated_frame)
            
            # Display frame (unless disabled)
            if not args.no_display and not args.batch:
                cv2.imshow('Enhanced Basketball Analysis', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    shot_detector.reset_session()
                    print("📊 Session reset")
            
            # Progress reporting
            if args.verbose and frame_count % 100 == 0:
                elapsed = time.time() - start_time
                processing_fps = frame_count / elapsed
                print(f"Frame {frame_count}: {processing_fps:.1f} FPS")
            
            # FPS limiting
            if args.fps_limit and not args.batch:
                time.sleep(max(0, 1/args.fps_limit - (time.time() - fps_counter)))
                
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        return 1
        
    finally:
        # Cleanup
        processing_time = time.time() - start_time
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 PROCESSING COMPLETE")
        print("=" * 50)
        print(f"⏱️ Total time: {processing_time:.2f}s")
        print(f"📹 Frames processed: {frame_count}")
        print(f"🚀 Average FPS: {frame_count / processing_time:.2f}")
        
        # Save results
        if args.save_json or args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            source_basename = Path(source_name).stem if not source_name.startswith('Camera') else f"camera_{args.camera}"
            results_file = os.path.join(args.output_dir, f"{source_basename}_analysis_{timestamp}.json")
            
            if save_analysis_results(shot_detector, results_file, model_info, processing_time):
                print(f"💾 Results saved: {results_file}")
        
        # Final statistics
        final_stats = shot_detector.get_shooting_statistics()
        if final_stats:
            print(f"\n🎯 FINAL STATISTICS")
            print("-" * 30)
            for key, value in final_stats.items():
                if isinstance(value, float):
                    if 'percentage' in key:
                        print(f"{key}: {value:.1f}%")
                    elif 'confidence' in key:
                        print(f"{key}: {value*100:.1f}%")
                    else:
                        print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")
        
        if args.benchmark:
            avg_fps = frame_count / processing_time
            print(f"\n⚡ PERFORMANCE BENCHMARK")
            print("-" * 30)
            print(f"Processing FPS: {avg_fps:.2f}")
            print(f"Frame time: {1000/avg_fps:.1f}ms")
            print(f"Efficiency: {(avg_fps/source_fps)*100:.1f}% of real-time")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 