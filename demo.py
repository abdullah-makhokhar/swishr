#!/usr/bin/env python3
"""
ShotSense Demo Script
Test the basketball shot analysis system with live camera or video file
"""

import cv2
import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from computer_vision.shot_detection import ShotDetector
from computer_vision.ball_detection import BallDetector  
from computer_vision.pose_estimation import PoseEstimator
from computer_vision.trajectory_analysis import TrajectoryAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ShotSense Basketball Analysis Demo')
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='0',
        help='Input source: camera index (0,1,2...) or video file path'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output video file path (optional)'
    )
    
    parser.add_argument(
        '--save-data',
        type=str,
        help='Save shot analysis data to JSON file'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='Minimum confidence threshold for ball detection'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Frame width for camera input'
    )
    
    parser.add_argument(
        '--height', 
        type=int,
        default=480,
        help='Frame height for camera input'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for video output'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without displaying video (for headless servers)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def setup_video_input(input_source: str, width: int, height: int):
    """Setup video input from camera or file"""
    try:
        # Try to convert to integer (camera index)
        camera_index = int(input_source)
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_index}")
            return None
            
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info(f"Using camera {camera_index} at {width}x{height}")
        return cap
        
    except ValueError:
        # Input is a file path
        if not os.path.exists(input_source):
            logger.error(f"Video file not found: {input_source}")
            return None
            
        cap = cv2.VideoCapture(input_source)
        
        if not cap.isOpened():
            logger.error(f"Could not open video file: {input_source}")
            return None
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"Using video file: {input_source} ({width}x{height} @ {fps}fps)")
        return cap


def setup_video_output(output_path: str, width: int, height: int, fps: int):
    """Setup video output writer"""
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Could not create output video: {output_path}")
            return None
            
        logger.info(f"Saving output to: {output_path}")
        return out
        
    except Exception as e:
        logger.error(f"Failed to setup video output: {e}")
        return None


def display_statistics(stats: dict, frame: cv2.Mat) -> cv2.Mat:
    """Display shooting statistics on frame"""
    try:
        if not stats:
            return frame
            
        # Create statistics overlay
        overlay = frame.copy()
        
        # Background for statistics
        stats_height = 200
        cv2.rectangle(overlay, (frame.shape[1] - 300, 0), 
                     (frame.shape[1], stats_height), (0, 0, 0), -1)
        
        # Apply transparency
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Add text
        y_offset = 25
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        
        cv2.putText(frame, "SHOOTING STATS", 
                   (frame.shape[1] - 290, y_offset), 
                   font, 0.6, (0, 255, 255), 2)
        y_offset += 30
        
        # Display key statistics
        key_stats = [
            ('Total Shots', 'total_shots'),
            ('Shooting %', 'shooting_percentage'),
            ('Avg Arc', 'avg_arc_angle'),
            ('Avg Form', 'avg_overall_form_score'),
            ('Confidence', 'avg_confidence')
        ]
        
        for label, key in key_stats:
            if key in stats:
                value = stats[key]
                if isinstance(value, float):
                    if key == 'shooting_percentage':
                        text = f"{label}: {value:.1f}%"
                    elif key in ['avg_arc_angle']:
                        text = f"{label}: {value:.1f}°"
                    else:
                        text = f"{label}: {value:.2f}"
                else:
                    text = f"{label}: {value}"
                    
                cv2.putText(frame, text, 
                           (frame.shape[1] - 290, y_offset),
                           font, font_scale, text_color, 1)
                y_offset += 20
                
        return frame
        
    except Exception as e:
        logger.error(f"Statistics display failed: {e}")
        return frame


def main():
    """Main demo function"""
    args = parse_arguments()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    logger.info("Starting ShotSense Basketball Analysis Demo")
    
    # Setup video input
    cap = setup_video_input(args.input, args.width, args.height)
    if cap is None:
        logger.error("Failed to setup video input")
        return 1
        
    # Get actual frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video output if specified
    out = None
    if args.output:
        out = setup_video_output(args.output, width, height, args.fps)
        
    # Initialize shot detection system
    logger.info("Initializing shot detection components...")
    
    try:
        # Initialize components with custom settings
        ball_detector = BallDetector(
            confidence_threshold=args.confidence_threshold
        )
        
        pose_estimator = PoseEstimator()
        
        trajectory_analyzer = TrajectoryAnalyzer()
        
        shot_detector = ShotDetector(
            ball_detector=ball_detector,
            pose_estimator=pose_estimator,
            trajectory_analyzer=trajectory_analyzer
        )
        
        logger.info("Shot detection system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize shot detection: {e}")
        return 1
        
    # Main processing loop
    frame_count = 0
    logger.info("Starting video processing... Press 'q' to quit, 'r' to reset session")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logger.info("End of video stream")
                break
                
            frame_count += 1
            
            # Process frame for shot detection
            current_shot, annotated_frame = shot_detector.process_frame(frame)
            
            # Add statistics overlay
            stats = shot_detector.get_shooting_statistics()
            annotated_frame = display_statistics(stats, annotated_frame)
            
            # Add frame counter
            cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write to output video if specified
            if out is not None:
                out.write(annotated_frame)
                
            # Display frame (unless no-display is set)
            if not args.no_display:
                cv2.imshow('ShotSense - Basketball Analysis', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('r'):
                    logger.info("Resetting session")
                    shot_detector.reset_session()
                elif key == ord('s') and args.save_data:
                    # Save current data
                    shot_detector.save_shot_data(args.save_data)
                    logger.info(f"Data saved to {args.save_data}")
                    
            # Log progress every 100 frames
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
                
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1
        
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        
        # Release video resources
        cap.release()
        if out is not None:
            out.release()
            
        if not args.no_display:
            cv2.destroyAllWindows()
            
        # Save final statistics and data
        final_stats = shot_detector.get_shooting_statistics()
        
        logger.info("=== FINAL SHOOTING STATISTICS ===")
        for key, value in final_stats.items():
            if isinstance(value, float):
                if 'percentage' in key:
                    logger.info(f"{key}: {value:.1f}%")
                elif 'angle' in key:
                    logger.info(f"{key}: {value:.1f}°")
                else:
                    logger.info(f"{key}: {value:.3f}")
            else:
                logger.info(f"{key}: {value}")
                
        # Save shot data if specified
        if args.save_data:
            success = shot_detector.save_shot_data(args.save_data)
            if success:
                logger.info(f"Shot analysis data saved to: {args.save_data}")
            else:
                logger.error("Failed to save shot data")
                
        logger.info("Demo completed successfully")
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 