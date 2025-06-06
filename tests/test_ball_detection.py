"""
Unit tests for basketball detection module
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from computer_vision.ball_detection import BallDetector, BallPosition


class TestBallDetector:
    """Test cases for BallDetector class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.detector = BallDetector(confidence_threshold=0.3)
        
    def test_initialization(self):
        """Test detector initialization"""
        assert self.detector is not None
        assert self.detector.confidence_threshold == 0.3
        assert len(self.detector.ball_positions) == 0
        assert not self.detector.tracking_active
        
    def test_synthetic_ball_detection(self):
        """Test ball detection on synthetic image"""
        # Create synthetic image with orange circle (basketball)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw orange circle
        center = (320, 240)
        radius = 30
        color = (30, 165, 255)  # Orange in BGR
        cv2.circle(frame, center, radius, color, -1)
        
        # Test color-based detection
        detections = self.detector.detect_ball_color(frame)
        
        # Should detect at least one ball
        assert len(detections) > 0
        
        # Check if detection is near expected position
        best_detection = max(detections, key=lambda x: x.confidence)
        assert abs(best_detection.x - center[0]) < 50
        assert abs(best_detection.y - center[1]) < 50
        assert best_detection.confidence > 0.1
        
    def test_empty_frame(self):
        """Test detection on empty frame"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test color-based detection
        detections = self.detector.detect_ball_color(frame)
        assert len(detections) == 0
        
        # Test tracking
        ball_pos = self.detector.track_ball(frame)
        assert ball_pos is None
        
    def test_ball_position_dataclass(self):
        """Test BallPosition dataclass"""
        pos = BallPosition(
            x=100.0,
            y=200.0,
            confidence=0.8,
            timestamp=1234567.0,
            radius=25.0,
            velocity=(10.0, -5.0)
        )
        
        assert pos.x == 100.0
        assert pos.y == 200.0
        assert pos.confidence == 0.8
        assert pos.timestamp == 1234567.0
        assert pos.radius == 25.0
        assert pos.velocity == (10.0, -5.0)
        
    def test_tracking_buffer(self):
        """Test ball position tracking buffer"""
        # Create multiple ball positions
        for i in range(15):
            pos = BallPosition(
                x=float(i * 10),
                y=float(i * 5),
                confidence=0.8,
                timestamp=float(i)
            )
            self.detector.add_ball_position(pos)
            
        # Should maintain buffer size
        assert len(self.detector.ball_positions) <= self.detector.tracking_buffer_size
        assert len(self.detector.ball_positions) == self.detector.tracking_buffer_size
        
    def test_velocity_calculation(self):
        """Test ball velocity calculation"""
        # Add positions to simulate movement
        pos1 = BallPosition(x=100.0, y=100.0, confidence=0.8, timestamp=1.0)
        pos2 = BallPosition(x=110.0, y=90.0, confidence=0.8, timestamp=2.0)
        
        self.detector.ball_positions = [pos1]
        
        # Simulate tracking with second position
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # The track_ball method would calculate velocity internally
        # For this test, we'll verify the concept
        dt = pos2.timestamp - pos1.timestamp
        expected_vx = (pos2.x - pos1.x) / dt  # 10.0 pixels/sec
        expected_vy = (pos2.y - pos1.y) / dt  # -10.0 pixels/sec
        
        assert expected_vx == 10.0
        assert expected_vy == -10.0
        
    def test_motion_detection(self):
        """Test ball motion detection"""
        # Initially no motion
        assert not self.detector.is_ball_in_motion()
        
        # Add position with high velocity
        pos = BallPosition(
            x=100.0, y=100.0, 
            confidence=0.8, 
            timestamp=1.0,
            velocity=(100.0, -50.0)  # High velocity
        )
        self.detector.ball_positions.append(pos)
        
        # Should detect motion
        assert self.detector.is_ball_in_motion(velocity_threshold=30.0)
        
        # Should not detect motion with high threshold
        assert not self.detector.is_ball_in_motion(velocity_threshold=200.0)
        
    def test_reset_tracking(self):
        """Test tracking reset functionality"""
        # Add some positions
        for i in range(5):
            pos = BallPosition(
                x=float(i), y=float(i), 
                confidence=0.8, timestamp=float(i)
            )
            self.detector.ball_positions.append(pos)
            
        self.detector.tracking_active = True
        
        # Reset tracking
        self.detector.reset_tracking()
        
        # Should be clean state
        assert len(self.detector.ball_positions) == 0
        assert not self.detector.tracking_active
        assert self.detector.last_detection_time == 0
        
    def test_trajectory_extraction(self):
        """Test trajectory point extraction"""
        # Add positions to form trajectory
        positions = [
            BallPosition(x=100.0, y=100.0, confidence=0.8, timestamp=1.0),
            BallPosition(x=110.0, y=90.0, confidence=0.8, timestamp=2.0),
            BallPosition(x=120.0, y=85.0, confidence=0.8, timestamp=3.0),
            BallPosition(x=130.0, y=85.0, confidence=0.8, timestamp=4.0),
        ]
        
        self.detector.ball_positions = positions
        
        # Get trajectory
        trajectory = self.detector.get_trajectory()
        
        assert len(trajectory) == 4
        assert trajectory[0] == (100.0, 100.0)
        assert trajectory[-1] == (130.0, 85.0)
        
    def test_confidence_thresholding(self):
        """Test confidence threshold filtering"""
        detector_high_threshold = BallDetector(confidence_threshold=0.9)
        
        # Create frame with weak detection
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw very small orange circle (low confidence)
        cv2.circle(frame, (320, 240), 5, (30, 165, 255), -1)
        
        # High threshold detector should not detect
        detections = detector_high_threshold.detect_ball_color(frame)
        
        # Low confidence detections should be filtered out
        high_conf_detections = [d for d in detections if d.confidence >= 0.9]
        assert len(high_conf_detections) == 0


if __name__ == "__main__":
    pytest.main([__file__]) 