"""
Basketball Detection and Tracking Module
Uses YOLO V8 for real-time basketball detection and tracking
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# COCO Dataset Class IDs for relevant objects
COCO_CLASSES = {
    'person': 0,
    'sports_ball': 32,  # This includes basketballs, footballs, etc.
    'ball': 32  # Alternative name for sports_ball
}

@dataclass
class BallPosition:
    """Data class for ball position and properties"""
    x: float
    y: float
    confidence: float
    timestamp: float
    radius: Optional[float] = None
    velocity: Optional[Tuple[float, float]] = None


class BallDetector:
    """
    Basketball detection and tracking using YOLO V8 and computer vision techniques
    
    Combines YOLO object detection with color-based tracking and Kalman filtering
    for robust ball detection in various lighting conditions.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 tracking_buffer_size: int = 10):
        """
        Initialize the ball detector
        
        Args:
            model_path: Path to custom YOLO model, uses default if None
            confidence_threshold: Minimum confidence for detections
            tracking_buffer_size: Number of frames to keep in tracking buffer
        """
        self.confidence_threshold = confidence_threshold
        self.tracking_buffer_size = tracking_buffer_size
        
        # Initialize YOLO model
        try:
            if model_path:
                self.yolo_model = YOLO(model_path)
            else:
                # Use YOLOv8n as default - will be fine-tuned for basketball
                self.yolo_model = YOLO('yolov8n.pt')
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
            
        # Ball tracking state
        self.ball_positions: List[BallPosition] = []
        self.tracking_active = False
        self.last_detection_time = 0
        
        # Color-based detection parameters for basketball
        # BGR Orange (30, 165, 255) converts to HSV (18, 225, 255)
        self.ball_color_ranges = [
            # Orange basketball HSV ranges - optimized for test color
            ([5, 50, 50], [25, 255, 255]),     # Orange range 1 (includes test color)
            ([160, 50, 50], [180, 255, 255]),  # Orange range 2 (wrap around for red-orange)
        ]
        
        # Kalman filter for smoothing
        self.kalman = cv2.KalmanFilter(4, 2)
        self._init_kalman_filter()
        
    def _init_kalman_filter(self):
        """Initialize Kalman filter for ball tracking"""
        # State transition matrix (position and velocity)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement noise covariance  
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # Error covariance
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
    def detect_ball_yolo(self, frame: np.ndarray) -> List[BallPosition]:
        """
        Detect basketball using YOLO V8
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detected ball positions
        """
        detections = []
        
        try:
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID and confidence
                        class_id = int(box.cls.cpu().numpy()[0])
                        confidence = float(box.conf.cpu().numpy()[0])
                        
                        # Only process sports_ball detections (not persons)
                        if class_id == COCO_CLASSES['sports_ball'] and confidence >= self.confidence_threshold:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                            
                            # Calculate center and radius
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            radius = max(x2 - x1, y2 - y1) / 2
                            
                            # Additional validation for basketball size and shape
                            width = x2 - x1
                            height = y2 - y1
                            aspect_ratio = width / height if height > 0 else 1.0
                            
                            # Filter out detections that are too elongated (not ball-like)
                            if 0.7 <= aspect_ratio <= 1.3:  # Nearly circular
                                ball_pos = BallPosition(
                                    x=center_x,
                                    y=center_y,
                                    confidence=confidence,
                                    timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
                                    radius=radius
                                )
                                detections.append(ball_pos)
                                
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            
        return detections
    
    def detect_ball_color(self, frame: np.ndarray) -> List[BallPosition]:
        """
        Detect basketball using color-based detection as fallback
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detected ball positions
        """
        detections = []
        
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for basketball colors
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for (lower, upper) in self.ball_color_ranges:
                color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask = cv2.bitwise_or(mask, color_mask)
            
            # Morphological operations to clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (basketball should be reasonably sized)
                if 100 < area < 5000:
                    # Get minimum enclosing circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.5:  # Reasonably circular
                            ball_pos = BallPosition(
                                x=float(x),
                                y=float(y),
                                confidence=min(circularity, 1.0),
                                timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
                                radius=float(radius)
                            )
                            detections.append(ball_pos)
                            
        except Exception as e:
            logger.error(f"Color-based detection failed: {e}")
            
        return detections
    
    def track_ball(self, frame: np.ndarray) -> Optional[BallPosition]:
        """
        Main ball tracking function combining YOLO and color detection
        
        Args:
            frame: Input video frame
            
        Returns:
            Best ball position estimate or None if no ball detected
        """
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Try YOLO detection first
        yolo_detections = self.detect_ball_yolo(frame)
        
        # If YOLO fails or low confidence, try color detection
        color_detections = []
        if not yolo_detections or all(det.confidence < 0.7 for det in yolo_detections):
            color_detections = self.detect_ball_color(frame)
        
        # Combine detections and select best one
        all_detections = yolo_detections + color_detections
        
        if not all_detections:
            return None
            
        # Select detection with highest confidence
        best_detection = max(all_detections, key=lambda x: x.confidence)
        
        # Update Kalman filter
        if self.ball_positions:
            # Predict next position
            prediction = self.kalman.predict()
            
            # Update with measurement
            measurement = np.array([[np.float32(best_detection.x)], 
                                  [np.float32(best_detection.y)]])
            self.kalman.correct(measurement)
            
            # Calculate velocity
            if len(self.ball_positions) > 0:
                dt = current_time - self.ball_positions[-1].timestamp
                if dt > 0:
                    vx = (best_detection.x - self.ball_positions[-1].x) / dt
                    vy = (best_detection.y - self.ball_positions[-1].y) / dt
                    best_detection.velocity = (vx, vy)
        else:
            # Initialize Kalman filter
            self.kalman.statePre = np.array([
                [np.float32(best_detection.x)],
                [np.float32(best_detection.y)],
                [0], [0]
            ])
            
        # Add to tracking buffer with automatic size management
        self.add_ball_position(best_detection)
            
        self.last_detection_time = current_time
        self.tracking_active = True
        
        return best_detection
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        """
        Get ball trajectory from recent positions
        
        Returns:
            List of (x, y) coordinates representing ball path
        """
        return [(pos.x, pos.y) for pos in self.ball_positions]
    
    def is_ball_in_motion(self, velocity_threshold: float = 50.0) -> bool:
        """
        Check if ball is currently in motion
        
        Args:
            velocity_threshold: Minimum velocity to consider ball in motion
            
        Returns:
            True if ball is moving above threshold
        """
        if not self.ball_positions or not self.ball_positions[-1].velocity:
            return False
            
        vx, vy = self.ball_positions[-1].velocity
        speed = np.sqrt(vx**2 + vy**2)
        
        return speed > velocity_threshold
    
    def _maintain_buffer_size(self):
        """Maintain the ball positions buffer size"""
        while len(self.ball_positions) > self.tracking_buffer_size:
            self.ball_positions.pop(0)
    
    def add_ball_position(self, ball_pos: BallPosition):
        """
        Add a ball position to tracking buffer with size management
        
        Args:
            ball_pos: Ball position to add
        """
        self.ball_positions.append(ball_pos)
        self._maintain_buffer_size()
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.ball_positions.clear()
        self.tracking_active = False
        self.last_detection_time = 0
        self._init_kalman_filter()
        
    def draw_detection(self, frame: np.ndarray, ball_pos: BallPosition) -> np.ndarray:
        """
        Draw ball detection on frame for visualization
        
        Args:
            frame: Input frame
            ball_pos: Ball position to draw
            
        Returns:
            Frame with detection overlay
        """
        if ball_pos.radius:
            # Draw circle
            cv2.circle(frame, 
                      (int(ball_pos.x), int(ball_pos.y)), 
                      int(ball_pos.radius), 
                      (0, 255, 0), 2)
            
            # Draw confidence
            cv2.putText(frame, 
                       f"Ball: {ball_pos.confidence:.2f}",
                       (int(ball_pos.x - ball_pos.radius), int(ball_pos.y - ball_pos.radius - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw velocity if available
            if ball_pos.velocity:
                vx, vy = ball_pos.velocity
                end_x = int(ball_pos.x + vx * 0.1)  # Scale for visualization
                end_y = int(ball_pos.y + vy * 0.1)
                cv2.arrowedLine(frame, 
                              (int(ball_pos.x), int(ball_pos.y)),
                              (end_x, end_y),
                              (255, 0, 0), 2)
        
        return frame 