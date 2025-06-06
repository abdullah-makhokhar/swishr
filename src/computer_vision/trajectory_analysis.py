"""
Basketball Shot Trajectory Analysis Module
Analyzes ball trajectory, arc angle, and shooting physics
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, NamedTuple
import logging
from dataclasses import dataclass
import math
from scipy import optimize
from scipy.interpolate import interp1d
from .ball_detection import BallPosition

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """Single point in ball trajectory"""
    x: float
    y: float
    timestamp: float
    velocity_x: Optional[float] = None
    velocity_y: Optional[float] = None
    acceleration_x: Optional[float] = None
    acceleration_y: Optional[float] = None


@dataclass
class ShotTrajectory:
    """Complete shot trajectory analysis"""
    points: List[TrajectoryPoint]
    release_point: Optional[TrajectoryPoint]
    apex_point: Optional[TrajectoryPoint]
    landing_point: Optional[TrajectoryPoint]
    arc_angle: Optional[float]  # degrees
    entry_angle: Optional[float]  # degrees
    peak_height: Optional[float]
    shot_distance: Optional[float]
    flight_time: Optional[float]
    initial_velocity: Optional[float]
    is_make: Optional[bool] = None
    confidence: float = 0.0


@dataclass
class PhysicsModel:
    """Basketball shot physics parameters"""
    gravity: float = 9.81  # m/s²
    ball_diameter: float = 0.239  # meters (official basketball)
    rim_height: float = 3.048  # meters (10 feet)
    rim_diameter: float = 0.457  # meters (18 inches)
    air_resistance: float = 0.1  # simplified drag coefficient


class TrajectoryAnalyzer:
    """
    Analyzes basketball shot trajectories for form and outcome prediction
    
    Calculates arc angles, entry angles, physics-based metrics, and predicts
    shot outcomes based on trajectory characteristics.
    """
    
    def __init__(self, 
                 min_trajectory_points: int = 5,
                 max_trajectory_gap: float = 0.5,  # seconds
                 court_calibration: Optional[Dict] = None):
        """
        Initialize trajectory analyzer
        
        Args:
            min_trajectory_points: Minimum points required for analysis
            max_trajectory_gap: Maximum time gap between points (seconds)
            court_calibration: Camera calibration parameters for real-world coordinates
        """
        self.min_trajectory_points = min_trajectory_points
        self.max_trajectory_gap = max_trajectory_gap
        self.court_calibration = court_calibration or {}
        
        # Physics model
        self.physics = PhysicsModel()
        
        # Trajectory tracking
        self.current_trajectory: List[TrajectoryPoint] = []
        self.completed_shots: List[ShotTrajectory] = []
        self.tracking_active = False
        
        # Calibration defaults (pixels to meters conversion)
        self.pixels_per_meter = self.court_calibration.get('pixels_per_meter', 100)
        self.camera_height = self.court_calibration.get('camera_height', 2.0)  # meters
        
    def add_ball_position(self, ball_pos: BallPosition) -> bool:
        """
        Add ball position to current trajectory
        
        Args:
            ball_pos: Ball position from detection
            
        Returns:
            True if position was added to trajectory
        """
        try:
            # Convert to trajectory point
            traj_point = TrajectoryPoint(
                x=ball_pos.x,
                y=ball_pos.y,
                timestamp=ball_pos.timestamp,
                velocity_x=ball_pos.velocity[0] if ball_pos.velocity else None,
                velocity_y=ball_pos.velocity[1] if ball_pos.velocity else None
            )
            
            # Check if this continues current trajectory or starts new one
            if self.current_trajectory:
                last_point = self.current_trajectory[-1]
                time_gap = traj_point.timestamp - last_point.timestamp
                
                if time_gap > self.max_trajectory_gap:
                    # Gap too large, finalize current trajectory and start new one
                    self._finalize_current_trajectory()
                    self.current_trajectory = [traj_point]
                else:
                    # Continue current trajectory
                    self.current_trajectory.append(traj_point)
            else:
                # Start new trajectory
                self.current_trajectory = [traj_point]
                self.tracking_active = True
                
            # Calculate velocities and accelerations
            self._calculate_kinematics()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add ball position to trajectory: {e}")
            return False
            
    def analyze_current_trajectory(self) -> Optional[ShotTrajectory]:
        """
        Analyze current trajectory in progress
        
        Returns:
            Partial trajectory analysis or None if insufficient data
        """
        if len(self.current_trajectory) < self.min_trajectory_points:
            return None
            
        try:
            return self._analyze_trajectory(self.current_trajectory.copy())
        except Exception as e:
            logger.error(f"Current trajectory analysis failed: {e}")
            return None
            
    def finalize_shot(self, is_make: Optional[bool] = None) -> Optional[ShotTrajectory]:
        """
        Finalize current shot trajectory
        
        Args:
            is_make: Whether shot was made (None if unknown)
            
        Returns:
            Complete trajectory analysis
        """
        if not self.current_trajectory:
            return None
            
        try:
            trajectory = self._finalize_current_trajectory()
            if trajectory and is_make is not None:
                trajectory.is_make = is_make
                
            return trajectory
            
        except Exception as e:
            logger.error(f"Shot finalization failed: {e}")
            return None
            
    def _finalize_current_trajectory(self) -> Optional[ShotTrajectory]:
        """Finalize and analyze current trajectory"""
        if len(self.current_trajectory) < self.min_trajectory_points:
            self.current_trajectory.clear()
            self.tracking_active = False
            return None
            
        try:
            trajectory = self._analyze_trajectory(self.current_trajectory)
            if trajectory:
                self.completed_shots.append(trajectory)
                
            self.current_trajectory.clear()
            self.tracking_active = False
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Trajectory finalization failed: {e}")
            self.current_trajectory.clear()
            self.tracking_active = False
            return None
            
    def _calculate_kinematics(self):
        """Calculate velocities and accelerations for trajectory points"""
        if len(self.current_trajectory) < 2:
            return
            
        try:
            # Calculate velocities
            for i in range(1, len(self.current_trajectory)):
                curr_point = self.current_trajectory[i]
                prev_point = self.current_trajectory[i-1]
                
                dt = curr_point.timestamp - prev_point.timestamp
                if dt > 0:
                    curr_point.velocity_x = (curr_point.x - prev_point.x) / dt
                    curr_point.velocity_y = (curr_point.y - prev_point.y) / dt
                    
            # Calculate accelerations
            if len(self.current_trajectory) >= 3:
                for i in range(2, len(self.current_trajectory)):
                    curr_point = self.current_trajectory[i]
                    prev_point = self.current_trajectory[i-1]
                    
                    if (curr_point.velocity_x is not None and prev_point.velocity_x is not None and
                        curr_point.velocity_y is not None and prev_point.velocity_y is not None):
                        
                        dt = curr_point.timestamp - prev_point.timestamp
                        if dt > 0:
                            curr_point.acceleration_x = (curr_point.velocity_x - prev_point.velocity_x) / dt
                            curr_point.acceleration_y = (curr_point.velocity_y - prev_point.velocity_y) / dt
                            
        except Exception as e:
            logger.error(f"Kinematics calculation failed: {e}")
            
    def _analyze_trajectory(self, points: List[TrajectoryPoint]) -> ShotTrajectory:
        """
        Perform comprehensive trajectory analysis
        
        Args:
            points: List of trajectory points
            
        Returns:
            Complete trajectory analysis
        """
        try:
            # Find key trajectory points
            release_point = self._find_release_point(points)
            apex_point = self._find_apex_point(points)
            landing_point = self._find_landing_point(points)
            
            # Calculate trajectory metrics
            arc_angle = self._calculate_arc_angle(points, release_point, apex_point)
            entry_angle = self._calculate_entry_angle(points, landing_point)
            peak_height = apex_point.y if apex_point else None
            shot_distance = self._calculate_shot_distance(release_point, landing_point)
            flight_time = self._calculate_flight_time(release_point, landing_point)
            initial_velocity = self._calculate_initial_velocity(points)
            
            # Calculate confidence based on trajectory quality
            confidence = self._calculate_trajectory_confidence(points)
            
            trajectory = ShotTrajectory(
                points=points,
                release_point=release_point,
                apex_point=apex_point,
                landing_point=landing_point,
                arc_angle=arc_angle,
                entry_angle=entry_angle,
                peak_height=peak_height,
                shot_distance=shot_distance,
                flight_time=flight_time,
                initial_velocity=initial_velocity,
                confidence=confidence
            )
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Trajectory analysis failed: {e}")
            return ShotTrajectory(points=points, confidence=0.0)
            
    def _find_release_point(self, points: List[TrajectoryPoint]) -> Optional[TrajectoryPoint]:
        """Find ball release point (typically first point with upward motion)"""
        try:
            for i, point in enumerate(points):
                if point.velocity_y is not None and point.velocity_y < 0:  # Negative y is upward in image coords
                    return point
                    
            # Fallback to first point
            return points[0] if points else None
            
        except Exception as e:
            logger.error(f"Release point detection failed: {e}")
            return points[0] if points else None
            
    def _find_apex_point(self, points: List[TrajectoryPoint]) -> Optional[TrajectoryPoint]:
        """Find trajectory apex (highest point)"""
        try:
            if not points:
                return None
                
            # Find point with minimum y-coordinate (top of arc)
            apex_point = min(points, key=lambda p: p.y)
            return apex_point
            
        except Exception as e:
            logger.error(f"Apex point detection failed: {e}")
            return None
            
    def _find_landing_point(self, points: List[TrajectoryPoint]) -> Optional[TrajectoryPoint]:
        """Find ball landing point (last detected point)"""
        try:
            return points[-1] if points else None
        except Exception as e:
            logger.error(f"Landing point detection failed: {e}")
            return None
            
    def _calculate_arc_angle(self, points: List[TrajectoryPoint], 
                           release_point: Optional[TrajectoryPoint],
                           apex_point: Optional[TrajectoryPoint]) -> Optional[float]:
        """Calculate shot arc angle in degrees"""
        try:
            if not release_point or not apex_point:
                return None
                
            # Calculate angle of trajectory at release
            dx = apex_point.x - release_point.x
            dy = release_point.y - apex_point.y  # Positive upward
            
            if dx == 0:
                return 90.0  # Vertical shot
                
            arc_angle = math.degrees(math.atan(dy / abs(dx)))
            return max(0, min(90, arc_angle))  # Clamp to 0-90 degrees
            
        except Exception as e:
            logger.error(f"Arc angle calculation failed: {e}")
            return None
            
    def _calculate_entry_angle(self, points: List[TrajectoryPoint], 
                             landing_point: Optional[TrajectoryPoint]) -> Optional[float]:
        """Calculate entry angle at rim"""
        try:
            if len(points) < 2 or not landing_point:
                return None
                
            # Use last few points to calculate entry trajectory
            end_points = points[-3:] if len(points) >= 3 else points[-2:]
            
            if len(end_points) < 2:
                return None
                
            # Calculate trajectory angle from last segment
            start_point = end_points[0]
            end_point = end_points[-1]
            
            dx = end_point.x - start_point.x
            dy = end_point.y - start_point.y
            
            if dx == 0:
                return 90.0
                
            entry_angle = math.degrees(math.atan(abs(dy) / abs(dx)))
            return max(0, min(90, entry_angle))
            
        except Exception as e:
            logger.error(f"Entry angle calculation failed: {e}")
            return None
            
    def _calculate_shot_distance(self, release_point: Optional[TrajectoryPoint],
                               landing_point: Optional[TrajectoryPoint]) -> Optional[float]:
        """Calculate horizontal shot distance in meters"""
        try:
            if not release_point or not landing_point:
                return None
                
            dx_pixels = abs(landing_point.x - release_point.x)
            distance_meters = dx_pixels / self.pixels_per_meter
            
            return distance_meters
            
        except Exception as e:
            logger.error(f"Shot distance calculation failed: {e}")
            return None
            
    def _calculate_flight_time(self, release_point: Optional[TrajectoryPoint],
                             landing_point: Optional[TrajectoryPoint]) -> Optional[float]:
        """Calculate ball flight time in seconds"""
        try:
            if not release_point or not landing_point:
                return None
                
            flight_time = landing_point.timestamp - release_point.timestamp
            return max(0, flight_time)
            
        except Exception as e:
            logger.error(f"Flight time calculation failed: {e}")
            return None
            
    def _calculate_initial_velocity(self, points: List[TrajectoryPoint]) -> Optional[float]:
        """Calculate initial velocity magnitude"""
        try:
            if len(points) < 2:
                return None
                
            # Use first few points with velocity data
            velocity_points = [p for p in points[:5] if p.velocity_x is not None and p.velocity_y is not None]
            
            if not velocity_points:
                return None
                
            # Average initial velocities
            vx_avg = sum(p.velocity_x for p in velocity_points) / len(velocity_points)
            vy_avg = sum(p.velocity_y for p in velocity_points) / len(velocity_points)
            
            # Convert pixel velocities to real velocities
            vx_real = vx_avg / self.pixels_per_meter
            vy_real = -vy_avg / self.pixels_per_meter  # Flip y-coordinate
            
            initial_velocity = math.sqrt(vx_real**2 + vy_real**2)
            return initial_velocity
            
        except Exception as e:
            logger.error(f"Initial velocity calculation failed: {e}")
            return None
            
    def _calculate_trajectory_confidence(self, points: List[TrajectoryPoint]) -> float:
        """Calculate confidence score for trajectory analysis"""
        try:
            if len(points) < self.min_trajectory_points:
                return 0.0
                
            confidence_factors = []
            
            # Factor 1: Number of points (more points = higher confidence)
            point_factor = min(1.0, len(points) / 15)  # Max confidence at 15+ points
            confidence_factors.append(point_factor)
            
            # Factor 2: Trajectory smoothness (less jitter = higher confidence)
            if len(points) >= 3:
                smoothness = self._calculate_smoothness(points)
                confidence_factors.append(smoothness)
                
            # Factor 3: Physics consistency (realistic trajectory = higher confidence)
            physics_consistency = self._check_physics_consistency(points)
            confidence_factors.append(physics_consistency)
            
            # Factor 4: Temporal consistency (even time intervals = higher confidence)
            temporal_consistency = self._check_temporal_consistency(points)
            confidence_factors.append(temporal_consistency)
            
            # Overall confidence is average of all factors
            overall_confidence = sum(confidence_factors) / len(confidence_factors)
            return max(0.0, min(1.0, overall_confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
            
    def _calculate_smoothness(self, points: List[TrajectoryPoint]) -> float:
        """Calculate trajectory smoothness score"""
        try:
            if len(points) < 3:
                return 0.0
                
            # Calculate direction changes
            direction_changes = []
            
            for i in range(1, len(points) - 1):
                p1, p2, p3 = points[i-1], points[i], points[i+1]
                
                # Vectors
                v1 = np.array([p2.x - p1.x, p2.y - p1.y])
                v2 = np.array([p3.x - p2.x, p3.y - p2.y])
                
                # Angle between vectors
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_change = np.degrees(np.arccos(cos_angle))
                    direction_changes.append(angle_change)
                    
            if not direction_changes:
                return 0.0
                
            # Lower average direction change = smoother trajectory
            avg_change = np.mean(direction_changes)
            smoothness = max(0, 1 - avg_change / 90)  # Normalize by max possible change
            
            return smoothness
            
        except Exception as e:
            logger.error(f"Smoothness calculation failed: {e}")
            return 0.0
            
    def _check_physics_consistency(self, points: List[TrajectoryPoint]) -> float:
        """Check if trajectory follows realistic physics"""
        try:
            if len(points) < 4:
                return 0.5  # Neutral score for insufficient data
                
            consistency_score = 0.0
            checks = 0
            
            # Check 1: Gravity effect (ball should decelerate upward, accelerate downward)
            points_with_accel = [p for p in points if p.acceleration_y is not None]
            if points_with_accel:
                # Acceleration should be roughly constant (gravity)
                accelerations = [p.acceleration_y for p in points_with_accel]
                accel_std = np.std(accelerations)
                gravity_consistency = max(0, 1 - accel_std / 500)  # Normalize by reasonable threshold
                consistency_score += gravity_consistency
                checks += 1
                
            # Check 2: Realistic velocities (not too fast or slow)
            points_with_vel = [p for p in points if p.velocity_x is not None and p.velocity_y is not None]
            if points_with_vel:
                speeds = [math.sqrt(p.velocity_x**2 + p.velocity_y**2) for p in points_with_vel]
                avg_speed = np.mean(speeds)
                
                # Reasonable speed range for basketball (in pixels/second)
                min_reasonable_speed = 50
                max_reasonable_speed = 1000
                
                if min_reasonable_speed <= avg_speed <= max_reasonable_speed:
                    speed_consistency = 1.0
                else:
                    speed_consistency = 0.5
                    
                consistency_score += speed_consistency
                checks += 1
                
            return consistency_score / checks if checks > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Physics consistency check failed: {e}")
            return 0.5
            
    def _check_temporal_consistency(self, points: List[TrajectoryPoint]) -> float:
        """Check temporal consistency of trajectory points"""
        try:
            if len(points) < 3:
                return 1.0  # Assume good for few points
                
            # Calculate time intervals
            time_intervals = []
            for i in range(1, len(points)):
                dt = points[i].timestamp - points[i-1].timestamp
                time_intervals.append(dt)
                
            # Check consistency of time intervals
            if time_intervals:
                interval_std = np.std(time_intervals)
                avg_interval = np.mean(time_intervals)
                
                if avg_interval > 0:
                    consistency = max(0, 1 - interval_std / avg_interval)
                else:
                    consistency = 0.0
                    
                return consistency
                
            return 0.0
            
        except Exception as e:
            logger.error(f"Temporal consistency check failed: {e}")
            return 0.5
            
    def predict_shot_outcome(self, trajectory: ShotTrajectory) -> Tuple[bool, float]:
        """
        Predict if shot will be made based on trajectory analysis
        
        Args:
            trajectory: Analyzed shot trajectory
            
        Returns:
            Tuple of (predicted_make, confidence)
        """
        try:
            if trajectory.confidence < 0.5:
                return False, 0.0
                
            prediction_factors = []
            
            # Factor 1: Arc angle (optimal 45-50 degrees)
            if trajectory.arc_angle is not None:
                optimal_arc = 47.5  # degrees
                arc_error = abs(trajectory.arc_angle - optimal_arc)
                arc_score = max(0, 1 - arc_error / 30)  # 30-degree tolerance
                prediction_factors.append(('arc', arc_score, 0.3))
                
            # Factor 2: Entry angle (optimal 30-50 degrees)
            if trajectory.entry_angle is not None:
                if 30 <= trajectory.entry_angle <= 50:
                    entry_score = 1.0
                else:
                    entry_error = min(abs(trajectory.entry_angle - 30), abs(trajectory.entry_angle - 50))
                    entry_score = max(0, 1 - entry_error / 20)
                prediction_factors.append(('entry', entry_score, 0.25))
                
            # Factor 3: Shot distance (closer shots more likely)
            if trajectory.shot_distance is not None:
                # Assume shots within 7 meters (3-point line) are reasonable
                if trajectory.shot_distance <= 7.0:
                    distance_score = 1.0 - (trajectory.shot_distance / 7.0) * 0.3
                else:
                    distance_score = 0.3  # Long shots less likely
                prediction_factors.append(('distance', distance_score, 0.2))
                
            # Factor 4: Trajectory smoothness (from confidence calculation)
            smoothness_score = trajectory.confidence
            prediction_factors.append(('smoothness', smoothness_score, 0.25))
            
            if not prediction_factors:
                return False, 0.0
                
            # Calculate weighted prediction
            weighted_score = 0.0
            total_weight = 0.0
            
            for factor_name, score, weight in prediction_factors:
                weighted_score += score * weight
                total_weight += weight
                
            final_score = weighted_score / total_weight if total_weight > 0 else 0.0
            
            # Prediction threshold
            make_threshold = 0.6
            predicted_make = final_score >= make_threshold
            
            return predicted_make, final_score
            
        except Exception as e:
            logger.error(f"Shot outcome prediction failed: {e}")
            return False, 0.0
            
    def get_shooting_metrics(self) -> Dict[str, float]:
        """Get overall shooting performance metrics"""
        try:
            if not self.completed_shots:
                return {}
                
            completed_with_outcome = [shot for shot in self.completed_shots if shot.is_make is not None]
            
            if not completed_with_outcome:
                return {}
                
            metrics = {}
            
            # Shooting percentage
            makes = sum(1 for shot in completed_with_outcome if shot.is_make)
            total_shots = len(completed_with_outcome)
            metrics['shooting_percentage'] = (makes / total_shots) * 100
            
            # Average arc angle
            arc_angles = [shot.arc_angle for shot in self.completed_shots if shot.arc_angle is not None]
            if arc_angles:
                metrics['avg_arc_angle'] = np.mean(arc_angles)
                metrics['arc_consistency'] = 1 - (np.std(arc_angles) / 20)  # Normalize by 20-degree range
                
            # Average entry angle
            entry_angles = [shot.entry_angle for shot in self.completed_shots if shot.entry_angle is not None]
            if entry_angles:
                metrics['avg_entry_angle'] = np.mean(entry_angles)
                
            # Shot distance analysis
            distances = [shot.shot_distance for shot in self.completed_shots if shot.shot_distance is not None]
            if distances:
                metrics['avg_shot_distance'] = np.mean(distances)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Shooting metrics calculation failed: {e}")
            return {}
            
    def draw_trajectory(self, frame: np.ndarray, trajectory: Optional[ShotTrajectory] = None) -> np.ndarray:
        """
        Draw trajectory on frame for visualization
        
        Args:
            frame: Input frame
            trajectory: Trajectory to draw (current if None)
            
        Returns:
            Frame with trajectory overlay
        """
        try:
            if trajectory is None:
                if not self.current_trajectory:
                    return frame
                points = self.current_trajectory
            else:
                points = trajectory.points
                
            if len(points) < 2:
                return frame
                
            # Draw trajectory path
            for i in range(1, len(points)):
                pt1 = (int(points[i-1].x), int(points[i-1].y))
                pt2 = (int(points[i].x), int(points[i].y))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)  # Yellow line
                
            # Draw key points
            if trajectory:
                # Release point
                if trajectory.release_point:
                    pt = (int(trajectory.release_point.x), int(trajectory.release_point.y))
                    cv2.circle(frame, pt, 8, (0, 255, 0), -1)  # Green circle
                    cv2.putText(frame, "Release", (pt[0] + 10, pt[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                # Apex point
                if trajectory.apex_point:
                    pt = (int(trajectory.apex_point.x), int(trajectory.apex_point.y))
                    cv2.circle(frame, pt, 8, (255, 0, 0), -1)  # Blue circle
                    cv2.putText(frame, "Apex", (pt[0] + 10, pt[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                # Landing point
                if trajectory.landing_point:
                    pt = (int(trajectory.landing_point.x), int(trajectory.landing_point.y))
                    cv2.circle(frame, pt, 8, (0, 0, 255), -1)  # Red circle
                    cv2.putText(frame, "Landing", (pt[0] + 10, pt[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                # Draw metrics
                y_offset = 30
                if trajectory.arc_angle is not None:
                    cv2.putText(frame, f"Arc: {trajectory.arc_angle:.1f}°", (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
                    
                if trajectory.entry_angle is not None:
                    cv2.putText(frame, f"Entry: {trajectory.entry_angle:.1f}°", (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
                    
                if trajectory.is_make is not None:
                    result_text = "MAKE" if trajectory.is_make else "MISS"
                    color = (0, 255, 0) if trajectory.is_make else (0, 0, 255)
                    cv2.putText(frame, result_text, (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
        except Exception as e:
            logger.error(f"Trajectory drawing failed: {e}")
            
        return frame
        
    def reset(self):
        """Reset analyzer state"""
        self.current_trajectory.clear()
        self.tracking_active = False 