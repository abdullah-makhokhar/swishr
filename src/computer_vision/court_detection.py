"""
Basketball Court Detection Module
Detects basketball court features including rim, court lines, and perspective calibration
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, NamedTuple
import logging
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class RimPosition:
    """Basketball rim position and properties"""
    center_x: float
    center_y: float
    radius: float
    confidence: float
    timestamp: float


@dataclass
class CourtFeatures:
    """Detected basketball court features"""
    rim_position: Optional[RimPosition]
    court_lines: List[Tuple[int, int, int, int]]  # Line segments (x1,y1,x2,y2)
    perspective_matrix: Optional[np.ndarray]
    court_bounds: Optional[Tuple[int, int, int, int]]  # (x1,y1,x2,y2)
    calibration_confidence: float


class CourtDetector:
    """
    Basketball court detection and calibration
    
    Detects court features like rim, lines, and provides perspective calibration
    for accurate distance and angle measurements.
    """
    
    def __init__(self, 
                 rim_detection_enabled: bool = True,
                 line_detection_enabled: bool = True,
                 calibration_enabled: bool = True):
        """
        Initialize court detector
        
        Args:
            rim_detection_enabled: Enable basketball rim detection
            line_detection_enabled: Enable court line detection
            calibration_enabled: Enable perspective calibration
        """
        self.rim_detection_enabled = rim_detection_enabled
        self.line_detection_enabled = line_detection_enabled
        self.calibration_enabled = calibration_enabled
        
        # Detection parameters
        self.rim_color_ranges = [
            # Orange rim HSV ranges
            ([5, 100, 100], [15, 255, 255]),
            ([170, 100, 100], [180, 255, 255]),
        ]
        
        # Court line detection (typically white lines)
        self.line_color_ranges = [
            ([0, 0, 200], [180, 30, 255]),  # White color range in HSV
        ]
        
        # Rim detection history for stability
        self.rim_history: List[RimPosition] = []
        self.max_rim_history = 10
        
        # Court calibration
        self.perspective_matrix = None
        self.inverse_perspective_matrix = None
        
        # Real-world court dimensions (in meters)
        self.court_length = 28.65  # NBA court length
        self.court_width = 15.24   # NBA court width
        self.rim_height = 3.048    # 10 feet
        self.rim_diameter = 0.457  # 18 inches
        
    def detect_court_features(self, frame: np.ndarray) -> CourtFeatures:
        """
        Detect all court features in frame
        
        Args:
            frame: Input video frame
            
        Returns:
            CourtFeatures object with detected features
        """
        try:
            rim_position = None
            court_lines = []
            perspective_matrix = None
            court_bounds = None
            calibration_confidence = 0.0
            
            # Detect rim if enabled
            if self.rim_detection_enabled:
                rim_position = self._detect_rim(frame)
                
            # Detect court lines if enabled
            if self.line_detection_enabled:
                court_lines = self._detect_court_lines(frame)
                
            # Perform calibration if enabled
            if self.calibration_enabled:
                perspective_matrix, court_bounds, calibration_confidence = self._calibrate_perspective(
                    frame, rim_position, court_lines)
                    
            return CourtFeatures(
                rim_position=rim_position,
                court_lines=court_lines,
                perspective_matrix=perspective_matrix,
                court_bounds=court_bounds,
                calibration_confidence=calibration_confidence
            )
            
        except Exception as e:
            logger.error(f"Court feature detection failed: {e}")
            return CourtFeatures(
                rim_position=None,
                court_lines=[],
                perspective_matrix=None,
                court_bounds=None,
                calibration_confidence=0.0
            )
            
    def _detect_rim(self, frame: np.ndarray) -> Optional[RimPosition]:
        """Detect basketball rim in frame"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for rim colors
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for (lower, upper) in self.rim_color_ranges:
                color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask = cv2.bitwise_or(mask, color_mask)
                
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rim_candidates = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (rim should be reasonable size)
                if 500 < area < 5000:
                    # Check if roughly circular
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.3:  # Reasonably circular
                            # Get enclosing circle
                            (x, y), radius = cv2.minEnclosingCircle(contour)
                            
                            # Check aspect ratio
                            rect = cv2.boundingRect(contour)
                            aspect_ratio = rect[2] / rect[3] if rect[3] > 0 else 0
                            
                            if 0.5 < aspect_ratio < 2.0:  # Reasonable aspect ratio
                                confidence = circularity * min(1.0, area / 2000)
                                
                                rim_candidates.append(RimPosition(
                                    center_x=float(x),
                                    center_y=float(y),
                                    radius=float(radius),
                                    confidence=confidence,
                                    timestamp=cv2.getTickCount() / cv2.getTickFrequency()
                                ))
                                
            if not rim_candidates:
                return None
                
            # Select best rim candidate
            best_rim = max(rim_candidates, key=lambda r: r.confidence)
            
            # Add to history and smooth
            self.rim_history.append(best_rim)
            if len(self.rim_history) > self.max_rim_history:
                self.rim_history.pop(0)
                
            # Return smoothed rim position
            return self._get_smoothed_rim_position()
            
        except Exception as e:
            logger.error(f"Rim detection failed: {e}")
            return None
            
    def _get_smoothed_rim_position(self) -> Optional[RimPosition]:
        """Get smoothed rim position from history"""
        try:
            if not self.rim_history:
                return None
                
            # Weight recent detections more heavily
            weights = np.linspace(0.5, 1.0, len(self.rim_history))
            
            # Weighted average of positions
            total_weight = np.sum(weights)
            
            avg_x = np.sum([rim.center_x * w for rim, w in zip(self.rim_history, weights)]) / total_weight
            avg_y = np.sum([rim.center_y * w for rim, w in zip(self.rim_history, weights)]) / total_weight
            avg_radius = np.sum([rim.radius * w for rim, w in zip(self.rim_history, weights)]) / total_weight
            avg_confidence = np.sum([rim.confidence * w for rim, w in zip(self.rim_history, weights)]) / total_weight
            
            return RimPosition(
                center_x=avg_x,
                center_y=avg_y,
                radius=avg_radius,
                confidence=avg_confidence,
                timestamp=self.rim_history[-1].timestamp
            )
            
        except Exception as e:
            logger.error(f"Rim position smoothing failed: {e}")
            return self.rim_history[-1] if self.rim_history else None
            
    def _detect_court_lines(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect basketball court lines"""
        try:
            # Convert to HSV for better color filtering
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for white lines
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for (lower, upper) in self.line_color_ranges:
                color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask = cv2.bitwise_or(mask, color_mask)
                
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Edge detection
            edges = cv2.Canny(mask, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=50,
                maxLineGap=10
            )
            
            court_lines = []
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Filter lines by length and angle
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    if length > 30:  # Minimum line length
                        # Calculate angle
                        angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                        angle = abs(angle)
                        
                        # Court lines are typically horizontal or vertical
                        if (angle < 15 or angle > 165 or  # Horizontal
                            75 < angle < 105):             # Vertical
                            court_lines.append((x1, y1, x2, y2))
                            
            return court_lines
            
        except Exception as e:
            logger.error(f"Court line detection failed: {e}")
            return []
            
    def _calibrate_perspective(self, frame: np.ndarray, 
                             rim_position: Optional[RimPosition],
                             court_lines: List[Tuple[int, int, int, int]]) -> Tuple[Optional[np.ndarray], Optional[Tuple], float]:
        """
        Calibrate perspective transformation for accurate measurements
        
        Returns:
            Tuple of (perspective_matrix, court_bounds, confidence)
        """
        try:
            h, w = frame.shape[:2]
            confidence = 0.0
            
            # Simple calibration based on rim position
            if rim_position:
                # Assume rim is at standard height and use it for basic calibration
                rim_x, rim_y = rim_position.center_x, rim_position.center_y
                
                # Estimate court bounds based on rim position
                # Rim is typically in upper portion of court view
                court_width_pixels = w * 0.8  # Assume court takes up 80% of frame width
                court_height_pixels = h * 0.6  # Assume court takes up 60% of frame height
                
                # Center court horizontally around rim
                court_left = max(0, int(rim_x - court_width_pixels / 2))
                court_right = min(w, int(rim_x + court_width_pixels / 2))
                court_top = max(0, int(rim_y - court_height_pixels * 0.3))  # Rim in upper portion
                court_bottom = min(h, int(rim_y + court_height_pixels * 0.7))
                
                court_bounds = (court_left, court_top, court_right, court_bottom)
                
                # Create basic perspective transformation
                # Map court bounds to standard court rectangle
                src_points = np.float32([
                    [court_left, court_top],
                    [court_right, court_top],
                    [court_right, court_bottom],
                    [court_left, court_bottom]
                ])
                
                # Destination points (standard court dimensions)
                dst_width = 400
                dst_height = int(dst_width * self.court_length / self.court_width)
                
                dst_points = np.float32([
                    [0, 0],
                    [dst_width, 0],
                    [dst_width, dst_height],
                    [0, dst_height]
                ])
                
                perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                self.perspective_matrix = perspective_matrix
                self.inverse_perspective_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
                
                confidence = rim_position.confidence * 0.7  # Basic calibration confidence
                
                return perspective_matrix, court_bounds, confidence
                
            # More sophisticated calibration using court lines (future enhancement)
            elif len(court_lines) >= 4:
                # Could implement more complex calibration using detected court lines
                # This would involve finding key court features like:
                # - Free throw line
                # - Three-point line
                # - Court boundaries
                # - Center court line
                
                # For now, return basic estimation
                court_bounds = (0, 0, w, h)
                confidence = 0.3
                
                return None, court_bounds, confidence
                
            else:
                return None, None, 0.0
                
        except Exception as e:
            logger.error(f"Perspective calibration failed: {e}")
            return None, None, 0.0
            
    def transform_point_to_court(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Transform pixel coordinates to court coordinates
        
        Args:
            point: (x, y) pixel coordinates
            
        Returns:
            (x, y) court coordinates in meters or None if no calibration
        """
        try:
            if self.perspective_matrix is None:
                return None
                
            # Apply perspective transformation
            pixel_point = np.array([[point]], dtype=np.float32)
            court_point = cv2.perspectiveTransform(pixel_point, self.perspective_matrix)
            
            # Convert from pixels to meters (assuming destination was in standard units)
            court_x = court_point[0][0][0] * self.court_width / 400  # Scale to real court width
            court_y = court_point[0][0][1] * self.court_length / (400 * self.court_length / self.court_width)
            
            return (court_x, court_y)
            
        except Exception as e:
            logger.error(f"Point transformation failed: {e}")
            return None
            
    def calculate_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> Optional[float]:
        """
        Calculate real-world distance between two pixel points
        
        Args:
            point1: First point (x, y) in pixels
            point2: Second point (x, y) in pixels
            
        Returns:
            Distance in meters or None if no calibration
        """
        try:
            court_point1 = self.transform_point_to_court(point1)
            court_point2 = self.transform_point_to_court(point2)
            
            if court_point1 is None or court_point2 is None:
                return None
                
            # Calculate Euclidean distance
            dx = court_point2[0] - court_point1[0]
            dy = court_point2[1] - court_point1[1]
            
            distance = math.sqrt(dx**2 + dy**2)
            return distance
            
        except Exception as e:
            logger.error(f"Distance calculation failed: {e}")
            return None
            
    def draw_court_features(self, frame: np.ndarray, features: CourtFeatures) -> np.ndarray:
        """
        Draw detected court features on frame
        
        Args:
            frame: Input frame
            features: Detected court features
            
        Returns:
            Frame with court features drawn
        """
        try:
            annotated_frame = frame.copy()
            
            # Draw rim position
            if features.rim_position:
                rim = features.rim_position
                center = (int(rim.center_x), int(rim.center_y))
                radius = int(rim.radius)
                
                # Draw rim circle
                cv2.circle(annotated_frame, center, radius, (0, 165, 255), 3)  # Orange color
                
                # Draw center point
                cv2.circle(annotated_frame, center, 5, (0, 165, 255), -1)
                
                # Draw confidence
                cv2.putText(annotated_frame, f"Rim: {rim.confidence:.2f}", 
                           (center[0] + radius + 10, center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                           
            # Draw court lines
            for line in features.court_lines:
                x1, y1, x2, y2 = line
                cv2.line(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                
            # Draw court bounds
            if features.court_bounds:
                x1, y1, x2, y2 = features.court_bounds
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw calibration confidence
                cv2.putText(annotated_frame, f"Calibration: {features.calibration_confidence:.2f}",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                           
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Court feature drawing failed: {e}")
            return frame
            
    def reset_calibration(self):
        """Reset calibration data"""
        self.rim_history.clear()
        self.perspective_matrix = None
        self.inverse_perspective_matrix = None
        
    def get_pixels_per_meter(self) -> Optional[float]:
        """
        Get current pixels per meter calibration factor
        
        Returns:
            Pixels per meter or None if no calibration
        """
        try:
            if self.perspective_matrix is None:
                return None
                
            # Use a standard distance to calculate scale
            # Transform two points 1 meter apart in court space back to pixels
            court_p1 = np.array([[[0, 0]]], dtype=np.float32)
            court_p2 = np.array([[[1, 0]]], dtype=np.float32)  # 1 meter right
            
            pixel_p1 = cv2.perspectiveTransform(court_p1, self.inverse_perspective_matrix)
            pixel_p2 = cv2.perspectiveTransform(court_p2, self.inverse_perspective_matrix)
            
            # Calculate pixel distance
            dx = pixel_p2[0][0][0] - pixel_p1[0][0][0]
            dy = pixel_p2[0][0][1] - pixel_p1[0][0][1]
            pixel_distance = math.sqrt(dx**2 + dy**2)
            
            return pixel_distance  # pixels per meter
            
        except Exception as e:
            logger.error(f"Pixels per meter calculation failed: {e}")
            return None 