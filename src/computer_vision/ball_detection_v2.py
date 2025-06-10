"""
Enhanced Ball Detection using Basketball-Specific YOLO Model
Maintains compatibility with existing ball_detection.py interface
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import time

from .models.basketball_yolo import BasketballYOLO

logger = logging.getLogger(__name__)

class EnhancedBallDetector:
    """Enhanced ball detector using basketball-specific YOLO model"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 conf_threshold: float = 0.3,
                 iou_threshold: float = 0.7,
                 fallback_to_color: bool = True,
                 enable_tracking: bool = True):
        """
        Initialize enhanced ball detector
        
        Args:
            model_path: Path to trained basketball model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            fallback_to_color: Whether to fallback to color detection if YOLO fails
            enable_tracking: Whether to enable object tracking
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.fallback_to_color = fallback_to_color
        self.enable_tracking = enable_tracking
        
        # Performance tracking
        self.detection_times = []
        self.detection_count = 0
        self.successful_detections = 0
        
        # Initialize basketball YOLO model
        try:
            self.basketball_model = BasketballYOLO(
                model_path=model_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                track_persist=enable_tracking
            )
            self.model_available = True
            logger.info("Basketball YOLO model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize basketball model: {e}")
            self.basketball_model = None
            self.model_available = False
            
        # Color-based detection parameters (fallback)
        self.color_ranges = {
            'orange': {
                'lower': np.array([8, 100, 100]),
                'upper': np.array([25, 255, 255])
            },
            'brown': {
                'lower': np.array([5, 50, 50]),
                'upper': np.array([15, 255, 200])
            },
            'red_orange': {
                'lower': np.array([0, 100, 100]),
                'upper': np.array([10, 255, 255])
            }
        }
        
    @property
    def model_path(self) -> Optional[str]:
        """Get the current model path"""
        if self.model_available and self.basketball_model:
            return self.basketball_model.model_path
        return None
        
    def detect_balls(self, frame: np.ndarray, use_tracking: bool = True) -> List[Dict[str, Any]]:
        """
        Detect basketballs in frame
        
        Args:
            frame: Input frame
            use_tracking: Whether to use tracking (if enabled)
            
        Returns:
            List of detected basketballs with metadata
        """
        start_time = time.time()
        detections = []
        
        try:
            if self.model_available:
                # Use basketball-specific YOLO model
                if use_tracking and self.enable_tracking:
                    results = self.basketball_model.detect_basketball_specific(frame)
                else:
                    yolo_results = self.basketball_model.detect(frame, classes=[0])  # Basketball only
                    results = self._parse_yolo_results(yolo_results)
                    
                # Extract basketball detections
                if results and 'basketballs' in results:
                    for detection in results['basketballs']:
                        ball_info = {
                            'bbox': detection['bbox'],  # [x1, y1, x2, y2]
                            'confidence': detection['confidence'],
                            'center': self._get_center_from_bbox(detection['bbox']),
                            'radius': self._estimate_radius_from_bbox(detection['bbox']),
                            'track_id': detection.get('track_id'),
                            'detection_method': 'yolo',
                            'aspect_ratio': detection.get('aspect_ratio', 1.0),
                            'size': detection.get('size', (0, 0))
                        }
                        detections.append(ball_info)
                        
                self.successful_detections += len(detections)
                
            # Fallback to color detection if no YOLO detections and fallback enabled
            if not detections and self.fallback_to_color:
                color_detections = self._detect_balls_by_color(frame)
                detections.extend(color_detections)
                
        except Exception as e:
            logger.error(f"Ball detection failed: {e}")
            # Fallback to color detection on error
            if self.fallback_to_color:
                detections = self._detect_balls_by_color(frame)
                
        # Update performance metrics
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        self.detection_count += 1
        
        # Keep only recent timing data
        if len(self.detection_times) > 100:
            self.detection_times = self.detection_times[-100:]
            
        return detections
        
    def _parse_yolo_results(self, results) -> Dict[str, Any]:
        """Parse YOLO results into basketball-specific format"""
        if not results or results.boxes is None:
            return {'basketballs': [], 'frame_info': {'basketball_count': 0}}
            
        basketballs = []
        boxes = results.boxes
        
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            
            if cls == 0:  # Basketball class
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                aspect_ratio = width / height if height > 0 else 0
                
                # Basic basketball validation
                if 0.7 <= aspect_ratio <= 1.3 and conf > self.conf_threshold:
                    detection = {
                        'bbox': bbox.tolist(),
                        'confidence': conf,
                        'track_id': None,
                        'class_id': cls,
                        'class_name': 'basketball',
                        'aspect_ratio': aspect_ratio,
                        'size': (width, height)
                    }
                    basketballs.append(detection)
                    
        return {
            'basketballs': basketballs,
            'frame_info': {'basketball_count': len(basketballs)}
        }
        
    def _detect_balls_by_color(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback color-based ball detection"""
        detections = []
        
        try:
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for basketball colors
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for color_name, color_range in self.color_ranges.items():
                color_mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
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
                if 500 < area < 50000:  # Adjust based on expected ball size
                    # Get bounding circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    # Validate circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.5:  # Reasonably circular
                            # Convert to bbox format
                            x1, y1 = int(x - radius), int(y - radius)
                            x2, y2 = int(x + radius), int(y + radius)
                            
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': min(0.8, circularity),  # Use circularity as confidence
                                'center': (int(x), int(y)),
                                'radius': int(radius),
                                'track_id': None,
                                'detection_method': 'color',
                                'aspect_ratio': 1.0,  # Assume circular
                                'size': (int(2*radius), int(2*radius)),
                                'circularity': circularity
                            }
                            detections.append(detection)
                            
        except Exception as e:
            logger.error(f"Color-based detection failed: {e}")
            
        return detections
        
    def _get_center_from_bbox(self, bbox: List[float]) -> Tuple[int, int]:
        """Get center point from bounding box"""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x, center_y)
        
    def _estimate_radius_from_bbox(self, bbox: List[float]) -> int:
        """Estimate radius from bounding box"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        # Use average of width and height as diameter
        radius = int((width + height) / 4)
        return max(radius, 1)
        
    def get_basketball_trajectory(self, track_id: int, window_size: int = 10) -> Optional[List[Tuple[float, float]]]:
        """Get trajectory for a specific basketball track"""
        if self.model_available and self.basketball_model:
            return self.basketball_model.get_basketball_trajectory(track_id, window_size)
        return None
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detection performance statistics"""
        stats = {
            'total_detections': self.detection_count,
            'successful_detections': self.successful_detections,
            'success_rate': self.successful_detections / self.detection_count if self.detection_count > 0 else 0,
            'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0,
            'model_available': self.model_available,
            'tracking_enabled': self.enable_tracking,
            'fallback_enabled': self.fallback_to_color
        }
        
        # Add model-specific stats if available
        if self.model_available and self.basketball_model:
            model_stats = self.basketball_model.get_performance_stats()
            stats.update({'model_stats': model_stats})
            
        return stats
        
    def reset_tracking(self):
        """Reset tracking history"""
        if self.model_available and self.basketball_model:
            self.basketball_model.reset_tracking()
            
        # Reset local stats
        self.detection_times.clear()
        self.detection_count = 0
        self.successful_detections = 0
        
    def update_thresholds(self, conf_threshold: float = None, iou_threshold: float = None):
        """Update detection thresholds"""
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
            if self.model_available and self.basketball_model:
                self.basketball_model.conf_threshold = conf_threshold
                
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
            if self.model_available and self.basketball_model:
                self.basketball_model.iou_threshold = iou_threshold
    
    def track_ball(self, frame: np.ndarray) -> Optional['BallPosition']:
        """
        Track basketball in frame - compatibility method for ShotDetector
        
        Args:
            frame: Input frame
            
        Returns:
            BallPosition object or None if no ball detected
        """
        try:
            detections = self.detect_balls(frame)
            
            if not detections:
                return None
                
            # Get the best detection (highest confidence)
            best_detection = max(detections, key=lambda d: d['confidence'])
            
            # Convert to BallPosition format
            from .ball_detection import BallPosition
            import cv2
            
            ball_pos = BallPosition(
                x=float(best_detection['center'][0]),
                y=float(best_detection['center'][1]),
                confidence=float(best_detection['confidence']),
                timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
                radius=float(best_detection['radius'])
            )
            
            # Add velocity if we have tracking info
            if best_detection.get('track_id') is not None and self.basketball_model:
                trajectory = self.basketball_model.get_basketball_trajectory(
                    best_detection['track_id'], window_size=3
                )
                if trajectory and len(trajectory) >= 2:
                    # Calculate velocity from recent positions
                    recent_positions = trajectory[-2:]
                    dx = recent_positions[1][0] - recent_positions[0][0]
                    dy = recent_positions[1][1] - recent_positions[0][1]
                    ball_pos.velocity = (dx, dy)
                    
            return ball_pos
            
        except Exception as e:
            logger.error(f"Ball tracking failed: {e}")
            return None
    
    def is_ball_in_motion(self, velocity_threshold: float = 50.0) -> bool:
        """
        Check if ball is currently in motion - compatibility method
        
        Args:
            velocity_threshold: Minimum velocity to consider as motion
            
        Returns:
            True if ball is in motion
        """
        try:
            if not self.basketball_model:
                return False
                
            # Check if we have any recent tracks with significant velocity
            for track_id, history in self.basketball_model.tracking_history.items():
                if len(history) >= 2:
                    recent = history[-2:]
                    dx = recent[1]['bbox'][0] - recent[0]['bbox'][0]
                    dy = recent[1]['bbox'][1] - recent[0]['bbox'][1]
                    velocity = np.sqrt(dx*dx + dy*dy)
                    
                    if velocity > velocity_threshold:
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"Motion detection failed: {e}")
            return False

    def draw_detection(self, frame: np.ndarray, ball_pos) -> np.ndarray:
        """
        Draw ball detection on frame for visualization - compatibility method for ShotDetector
        
        Args:
            frame: Input frame
            ball_pos: BallPosition object to draw
            
        Returns:
            Frame with detection overlay
        """
        try:
            annotated_frame = frame.copy()
            
            if ball_pos and hasattr(ball_pos, 'x') and hasattr(ball_pos, 'y'):
                x, y = int(ball_pos.x), int(ball_pos.y)
                
                # Get radius - try different attribute names for compatibility
                radius = 15  # Default radius
                if hasattr(ball_pos, 'radius') and ball_pos.radius:
                    radius = int(ball_pos.radius)
                elif hasattr(ball_pos, 'size') and ball_pos.size:
                    radius = int(ball_pos.size / 2)
                
                # Draw main detection circle
                cv2.circle(annotated_frame, (x, y), radius, (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                
                # Draw confidence if available
                if hasattr(ball_pos, 'confidence'):
                    confidence_text = f"Ball: {ball_pos.confidence:.2f}"
                    cv2.putText(annotated_frame, confidence_text,
                              (x - radius, y - radius - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw velocity arrow if available
                if hasattr(ball_pos, 'velocity') and ball_pos.velocity:
                    vx, vy = ball_pos.velocity
                    # Scale velocity for visualization
                    scale = 0.1
                    end_x = int(x + vx * scale)
                    end_y = int(y + vy * scale)
                    cv2.arrowedLine(annotated_frame, (x, y), (end_x, end_y), (255, 0, 0), 2)
                
                # Draw tracking ID if available
                if hasattr(ball_pos, 'track_id') and ball_pos.track_id is not None:
                    track_text = f"ID: {ball_pos.track_id}"
                    cv2.putText(annotated_frame, track_text,
                              (x + radius + 5, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Draw detection method if this is an enhanced detection
                if hasattr(ball_pos, 'detection_method'):
                    method_text = f"Method: {ball_pos.detection_method}"
                    cv2.putText(annotated_frame, method_text,
                              (x - radius, y + radius + 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Draw detection failed: {e}")
            return frame

    def draw_enhanced_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw multiple ball detections with enhanced information
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries from detect_balls()
            
        Returns:
            Frame with all detections annotated
        """
        try:
            annotated_frame = frame.copy()
            
            for i, detection in enumerate(detections):
                center = detection.get('center', (0, 0))
                radius = detection.get('radius', 15)
                confidence = detection.get('confidence', 0.0)
                method = detection.get('detection_method', 'unknown')
                track_id = detection.get('track_id')
                
                x, y = int(center[0]), int(center[1])
                
                # Color coding by detection method
                if method == 'yolo':
                    color = (0, 255, 0)  # Green for YOLO
                elif method == 'color':
                    color = (0, 255, 255)  # Yellow for color-based
                else:
                    color = (128, 128, 128)  # Gray for unknown
                
                # Draw detection circle
                cv2.circle(annotated_frame, (x, y), int(radius), color, 2)
                cv2.circle(annotated_frame, (x, y), 3, color, -1)
                
                # Draw confidence
                conf_text = f"{confidence:.2f}"
                cv2.putText(annotated_frame, conf_text,
                          (x - radius, y - radius - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw detection index
                cv2.putText(annotated_frame, f"#{i}",
                          (x + radius + 5, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw tracking ID if available
                if track_id is not None:
                    track_text = f"T:{track_id}"
                    cv2.putText(annotated_frame, track_text,
                              (x + radius + 5, y + 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Draw method indicator
                method_short = method[:4].upper()
                cv2.putText(annotated_frame, method_short,
                          (x - radius, y + radius + 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw detection summary
            if detections:
                summary_text = f"Detections: {len(detections)}"
                cv2.putText(annotated_frame, summary_text,
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Enhanced detection drawing failed: {e}")
            return frame


# Compatibility function for existing code
def detect_basketball(frame: np.ndarray, 
                     detector: Optional[EnhancedBallDetector] = None,
                     **kwargs) -> List[Tuple[int, int, int]]:
    """
    Compatibility function for existing ball detection interface
    
    Args:
        frame: Input frame
        detector: Optional detector instance (will create if None)
        **kwargs: Additional arguments
        
    Returns:
        List of (x, y, radius) tuples for detected balls
    """
    if detector is None:
        detector = EnhancedBallDetector()
        
    detections = detector.detect_balls(frame)
    
    # Convert to legacy format
    legacy_detections = []
    for detection in detections:
        center = detection['center']
        radius = detection['radius']
        legacy_detections.append((center[0], center[1], radius))
        
    return legacy_detections 