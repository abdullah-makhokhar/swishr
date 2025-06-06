"""
Basketball Shot Detection Module
Combines ball detection, pose estimation, and trajectory analysis
to detect and analyze complete shooting motions
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, NamedTuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

from .ball_detection import BallDetector, BallPosition
from .pose_estimation import PoseEstimator, ShootingPose
from .trajectory_analysis import TrajectoryAnalyzer, ShotTrajectory

logger = logging.getLogger(__name__)


class ShotPhase(Enum):
    """Different phases of a basketball shot"""
    IDLE = "idle"
    SETUP = "setup"  # Player preparing to shoot
    SHOOTING = "shooting"  # Active shooting motion
    BALL_FLIGHT = "ball_flight"  # Ball in the air
    COMPLETED = "completed"  # Shot finished


@dataclass 
class FormAnalysis:
    """Basketball shooting form analysis results"""
    overall_score: float = 0.0
    elbow_alignment: float = 0.0
    follow_through: float = 0.0
    balance: float = 0.0
    arc_consistency: float = 0.0
    release_timing: float = 0.0
    confidence: float = 0.0


@dataclass
class ShotEvent:
    """Complete basketball shot event"""
    shot_id: str = ""
    start_time: float = 0.0
    end_time: Optional[float] = None
    phase: ShotPhase = ShotPhase.IDLE
    shooter_poses: List[ShootingPose] = None
    ball_positions: List[BallPosition] = None
    trajectory: Optional[ShotTrajectory] = None
    shot_successful: bool = False  # Renamed from is_make for consistency
    form_analysis: Optional[FormAnalysis] = None
    detection_confidence: float = 0.0  # Renamed from confidence for clarity
    timestamp: Optional[datetime] = None
    release_point: Optional[BallPosition] = None
    
    def __post_init__(self):
        if self.shooter_poses is None:
            self.shooter_poses = []
        if self.ball_positions is None:
            self.ball_positions = []


class ShotDetector:
    """
    Main shot detection class that orchestrates all computer vision components
    
    Detects complete basketball shooting sequences by combining:
    - Ball detection and tracking
    - Player pose estimation  
    - Trajectory analysis
    - Shot outcome prediction
    """
    
    def __init__(self,
                 ball_detector: Optional[BallDetector] = None,
                 pose_estimator: Optional[PoseEstimator] = None,
                 trajectory_analyzer: Optional[TrajectoryAnalyzer] = None,
                 shot_timeout: float = 5.0):
        """
        Initialize shot detector with computer vision components
        
        Args:
            ball_detector: Ball detection component
            pose_estimator: Pose estimation component  
            trajectory_analyzer: Trajectory analysis component
            shot_timeout: Time to wait before finalizing shot (seconds)
        """
        # Initialize components with defaults if not provided
        self.ball_detector = ball_detector or BallDetector()
        self.pose_estimator = pose_estimator or PoseEstimator()
        self.trajectory_analyzer = trajectory_analyzer or TrajectoryAnalyzer()
        
        self.shot_timeout = shot_timeout
        
        # Shot detection state
        self.current_shot: Optional[ShotEvent] = None
        self.completed_shots: List[ShotEvent] = []
        self.shot_counter = 0
        
        # Detection parameters
        self.min_shooting_duration = 0.5  # Minimum time for valid shot
        self.ball_lost_timeout = 1.0  # Time before declaring ball lost
        
        # State tracking
        self.last_ball_time = 0.0
        self.last_pose_time = 0.0
        self.last_frame_time = 0.0
        
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[ShotEvent], np.ndarray]:
        """
        Process video frame and detect basketball shots
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (current_shot_event, annotated_frame)
        """
        try:
            current_time = time.time()
            self.last_frame_time = current_time
            
            # Track ball in frame
            ball_position = self.ball_detector.track_ball(frame)
            if ball_position:
                self.last_ball_time = current_time
                
            # Detect pose in frame
            pose = self.pose_estimator.detect_pose(frame)
            if pose:
                self.last_pose_time = current_time
                
            # Update shot state machine
            self._update_shot_state(ball_position, pose, current_time)
            
            # Add data to current shot if active
            if self.current_shot:
                if ball_position:
                    self.current_shot.ball_positions.append(ball_position)
                    self.trajectory_analyzer.add_ball_position(ball_position)
                    
                if pose:
                    self.current_shot.shooter_poses.append(pose)
                    
            # Check for shot completion
            self._check_shot_completion(current_time)
            
            # Create annotated frame
            annotated_frame = self._annotate_frame(frame, ball_position, pose)
            
            return self.current_shot, annotated_frame
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return None, frame
            
    def _update_shot_state(self, ball_position: Optional[BallPosition], 
                          pose: Optional[ShootingPose], current_time: float):
        """Update shot detection state machine"""
        try:
            if self.current_shot is None:
                # Check for new shot initiation
                if self._detect_shot_start(ball_position, pose):
                    self._start_new_shot(current_time)
                    
            else:
                # Update existing shot phase
                self._update_shot_phase(ball_position, pose, current_time)
                
        except Exception as e:
            logger.error(f"Shot state update failed: {e}")
            
    def _detect_shot_start(self, ball_position: Optional[BallPosition], 
                          pose: Optional[ShootingPose]) -> bool:
        """Detect start of new shooting motion"""
        try:
            # Require both pose and ball detection
            if not pose or not ball_position:
                return False
                
            # Check if player is in shooting stance
            if not pose.is_shooting_stance:
                return False
                
            # Check if ball is in motion (being shot)
            if not self.ball_detector.is_ball_in_motion():
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Shot start detection failed: {e}")
            return False
            
    def _start_new_shot(self, current_time: float):
        """Initialize new shot event"""
        try:
            self.shot_counter += 1
            shot_id = f"shot_{self.shot_counter:04d}_{int(current_time)}"
            
            self.current_shot = ShotEvent(
                shot_id=shot_id,
                start_time=current_time,
                end_time=None,
                phase=ShotPhase.SETUP,
                shooter_poses=[],
                ball_positions=[],
                trajectory=None,
                shot_successful=False,
                form_analysis=None,
                detection_confidence=0.0,
                timestamp=datetime.now()
            )
            
            # Reset analyzers
            self.trajectory_analyzer.reset()
            
            logger.info(f"Started new shot: {shot_id}")
            
        except Exception as e:
            logger.error(f"Shot initialization failed: {e}")
            
    def _update_shot_phase(self, ball_position: Optional[BallPosition], 
                          pose: Optional[ShootingPose], current_time: float):
        """Update current shot phase based on observations"""
        try:
            if not self.current_shot:
                return
                
            current_phase = self.current_shot.phase
            
            if current_phase == ShotPhase.SETUP:
                # Check for transition to shooting
                if ball_position and self.ball_detector.is_ball_in_motion():
                    self.current_shot.phase = ShotPhase.SHOOTING
                    logger.debug(f"Shot {self.current_shot.shot_id}: SETUP -> SHOOTING")
                    
            elif current_phase == ShotPhase.SHOOTING:
                # Check for transition to ball flight
                if pose and not pose.is_shooting_stance:
                    # Player no longer in shooting stance - ball released
                    self.current_shot.phase = ShotPhase.BALL_FLIGHT
                    logger.debug(f"Shot {self.current_shot.shot_id}: SHOOTING -> BALL_FLIGHT")
                    
            elif current_phase == ShotPhase.BALL_FLIGHT:
                # Check for shot completion
                time_since_ball = current_time - self.last_ball_time
                if time_since_ball > self.ball_lost_timeout:
                    # Ball no longer detected - shot completed
                    self.current_shot.phase = ShotPhase.COMPLETED
                    logger.debug(f"Shot {self.current_shot.shot_id}: BALL_FLIGHT -> COMPLETED")
                    
        except Exception as e:
            logger.error(f"Shot phase update failed: {e}")
            
    def _check_shot_completion(self, current_time: float):
        """Check if current shot should be finalized"""
        try:
            if not self.current_shot:
                return
                
            should_complete = False
            completion_reason = ""
            
            # Reason 1: Shot phase indicates completion
            if self.current_shot.phase == ShotPhase.COMPLETED:
                should_complete = True
                completion_reason = "phase_completed"
                
            # Reason 2: Timeout reached
            elif current_time - self.current_shot.start_time > self.shot_timeout:
                should_complete = True
                completion_reason = "timeout"
                
            # Reason 3: Minimum duration met and ball lost
            elif (current_time - self.current_shot.start_time > self.min_shooting_duration and
                  current_time - self.last_ball_time > self.ball_lost_timeout):
                should_complete = True
                completion_reason = "ball_lost"
                
            if should_complete:
                self._finalize_shot(current_time, completion_reason)
                
        except Exception as e:
            logger.error(f"Shot completion check failed: {e}")
            
    def _finalize_shot(self, current_time: float, reason: str):
        """Finalize and analyze completed shot"""
        try:
            if not self.current_shot:
                return
                
            # Set end time
            self.current_shot.end_time = current_time
            
            # Finalize trajectory analysis
            trajectory = self.trajectory_analyzer.finalize_shot()
            self.current_shot.trajectory = trajectory
            
            # Analyze shooting form
            if self.current_shot.shooter_poses:
                self.current_shot.form_analysis = self._analyze_shooting_form()
                
            # Predict shot outcome if trajectory available
            if trajectory:
                predicted_make, prediction_confidence = self.trajectory_analyzer.predict_shot_outcome(trajectory)
                if prediction_confidence > 0.5:  # Only use prediction if confident
                    self.current_shot.shot_successful = predicted_make
                    
            # Calculate overall confidence
            self.current_shot.detection_confidence = self._calculate_shot_confidence()
            
            # Add to completed shots
            self.completed_shots.append(self.current_shot)
            
            logger.info(f"Finalized shot {self.current_shot.shot_id} (reason: {reason}, "
                       f"confidence: {self.current_shot.detection_confidence:.2f})")
            
            # Reset for next shot
            self.current_shot = None
            
        except Exception as e:
            logger.error(f"Shot finalization failed: {e}")
            self.current_shot = None
            
    def _analyze_shooting_form(self) -> FormAnalysis:
        """Analyze shooting form from collected pose data"""
        try:
            if not self.current_shot or not self.current_shot.shooter_poses:
                return FormAnalysis()
                
            # Find shooting poses (poses during shooting phase)
            shooting_poses = []
            for pose in self.current_shot.shooter_poses:
                if pose.is_shooting_stance:
                    shooting_poses.append(pose)
                    
            if not shooting_poses:
                return FormAnalysis()
                
            # Analyze each shooting pose and aggregate metrics
            elbow_scores = []
            follow_through_scores = []
            balance_scores = []
            arc_scores = []
            timing_scores = []
            
            for pose in shooting_poses:
                # Calculate individual metrics
                elbow_score = self._calculate_elbow_alignment(pose)
                follow_through_score = self._calculate_follow_through(pose)
                balance_score = self._calculate_balance(pose)
                
                elbow_scores.append(elbow_score)
                follow_through_scores.append(follow_through_score)
                balance_scores.append(balance_score)
                
            # Calculate averages
            elbow_avg = sum(elbow_scores) / len(elbow_scores) if elbow_scores else 0.0
            follow_through_avg = sum(follow_through_scores) / len(follow_through_scores) if follow_through_scores else 0.0
            balance_avg = sum(balance_scores) / len(balance_scores) if balance_scores else 0.0
            
            # Calculate arc consistency from trajectory
            arc_consistency = 0.0
            if self.current_shot.trajectory:
                ideal_arc = 45.0  # Optimal arc angle
                actual_arc = self.current_shot.trajectory.arc_angle
                arc_consistency = max(0.0, 1.0 - abs(actual_arc - ideal_arc) / ideal_arc)
                
            # Calculate release timing (simplified)
            release_timing = 0.8  # Default good timing
            
            # Calculate overall score
            overall_score = (elbow_avg * 0.25 + follow_through_avg * 0.25 + 
                           balance_avg * 0.30 + arc_consistency * 0.20)
            
            # Calculate confidence based on data quality
            confidence = min(1.0, len(shooting_poses) / 10.0)  # More poses = higher confidence
            
            return FormAnalysis(
                overall_score=overall_score,
                elbow_alignment=elbow_avg,
                follow_through=follow_through_avg,
                balance=balance_avg,
                arc_consistency=arc_consistency,
                release_timing=release_timing,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Shooting form analysis failed: {e}")
            return FormAnalysis()
            
    def _calculate_elbow_alignment(self, pose: ShootingPose) -> float:
        """Calculate elbow alignment score"""
        try:
            # Simplified calculation - in practice would use joint angles
            return 0.8  # Placeholder
        except:
            return 0.0
            
    def _calculate_follow_through(self, pose: ShootingPose) -> float:
        """Calculate follow through score"""
        try:
            # Simplified calculation - in practice would analyze wrist position
            return 0.75  # Placeholder
        except:
            return 0.0
            
    def _calculate_balance(self, pose: ShootingPose) -> float:
        """Calculate balance score"""
        try:
            # Simplified calculation - in practice would analyze foot positioning
            return 0.85  # Placeholder
        except:
            return 0.0
            
    def _calculate_shot_confidence(self) -> float:
        """Calculate overall confidence in shot detection and analysis"""
        try:
            if not self.current_shot:
                return 0.0
                
            confidence_factors = []
            
            # Factor 1: Trajectory confidence
            if self.current_shot.trajectory:
                trajectory_confidence = self.current_shot.trajectory.confidence
                confidence_factors.append(trajectory_confidence)
                
            # Factor 2: Pose detection quality
            if self.current_shot.shooter_poses:
                pose_confidences = [pose.confidence for pose in self.current_shot.shooter_poses]
                avg_pose_confidence = sum(pose_confidences) / len(pose_confidences)
                confidence_factors.append(avg_pose_confidence)
                
            # Factor 3: Ball detection consistency
            if self.current_shot.ball_positions:
                ball_confidences = [ball.confidence for ball in self.current_shot.ball_positions]
                avg_ball_confidence = sum(ball_confidences) / len(ball_confidences)
                confidence_factors.append(avg_ball_confidence)
                
            # Factor 4: Shot duration (reasonable duration increases confidence)
            if self.current_shot.end_time:
                duration = self.current_shot.end_time - self.current_shot.start_time
                if 0.5 <= duration <= 3.0:  # Reasonable shot duration
                    duration_confidence = 1.0
                else:
                    duration_confidence = 0.5
                confidence_factors.append(duration_confidence)
                
            if not confidence_factors:
                return 0.0
                
            # Overall confidence is average of all factors
            overall_confidence = sum(confidence_factors) / len(confidence_factors)
            return max(0.0, min(1.0, overall_confidence))
            
        except Exception as e:
            logger.error(f"Shot confidence calculation failed: {e}")
            return 0.0
            
    def _annotate_frame(self, frame: np.ndarray, 
                       ball_position: Optional[BallPosition],
                       pose: Optional[ShootingPose]) -> np.ndarray:
        """Add visual annotations to frame"""
        try:
            annotated_frame = frame.copy()
            
            # Draw ball detection
            if ball_position:
                annotated_frame = self.ball_detector.draw_detection(annotated_frame, ball_position)
                
            # Draw pose estimation
            if pose:
                annotated_frame = self.pose_estimator.draw_pose(annotated_frame, pose)
                
            # Draw trajectory
            if self.current_shot and self.current_shot.trajectory:
                annotated_frame = self.trajectory_analyzer.draw_trajectory(
                    annotated_frame, self.current_shot.trajectory)
            elif self.trajectory_analyzer.current_trajectory:
                annotated_frame = self.trajectory_analyzer.draw_trajectory(annotated_frame)
                
            # Draw shot status
            if self.current_shot:
                status_text = f"Shot: {self.current_shot.phase.value.upper()}"
                cv2.putText(annotated_frame, status_text, (10, frame.shape[0] - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                           
                duration = self.last_frame_time - self.current_shot.start_time
                duration_text = f"Duration: {duration:.1f}s"
                cv2.putText(annotated_frame, duration_text, (10, frame.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                           
            # Draw shot statistics
            stats_text = f"Completed Shots: {len(self.completed_shots)}"
            cv2.putText(annotated_frame, stats_text, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                       
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Frame annotation failed: {e}")
            return frame
            
    def get_shooting_statistics(self) -> Dict[str, float]:
        """Get comprehensive shooting statistics"""
        try:
            if not self.completed_shots:
                return {}
                
            stats = {}
            
            # Basic statistics
            total_shots = len(self.completed_shots)
            stats['total_shots'] = total_shots
            
            # Shots with known outcomes
            shots_with_outcome = [shot for shot in self.completed_shots if shot.shot_successful is not None]
            if shots_with_outcome:
                makes = sum(1 for shot in shots_with_outcome if shot.shot_successful)
                stats['shooting_percentage'] = (makes / len(shots_with_outcome)) * 100
                stats['makes'] = makes
                stats['misses'] = len(shots_with_outcome) - makes
                
            # Form analysis aggregation
            shots_with_form = [shot for shot in self.completed_shots if shot.form_analysis]
            
            if shots_with_form:
                overall_scores = [shot.form_analysis.overall_score for shot in shots_with_form]
                elbow_scores = [shot.form_analysis.elbow_alignment for shot in shots_with_form]
                follow_through_scores = [shot.form_analysis.follow_through for shot in shots_with_form]
                balance_scores = [shot.form_analysis.balance for shot in shots_with_form]
                arc_scores = [shot.form_analysis.arc_consistency for shot in shots_with_form]
                timing_scores = [shot.form_analysis.release_timing for shot in shots_with_form]
                
                stats.update({
                    'avg_overall_form_score': sum(overall_scores) / len(overall_scores),
                    'avg_elbow_alignment': sum(elbow_scores) / len(elbow_scores),
                    'avg_follow_through': sum(follow_through_scores) / len(follow_through_scores),
                    'avg_balance': sum(balance_scores) / len(balance_scores),
                    'avg_arc_consistency': sum(arc_scores) / len(arc_scores),
                    'avg_release_timing': sum(timing_scores) / len(timing_scores)
                })
                
            # Trajectory statistics
            trajectory_stats = self.trajectory_analyzer.get_shooting_metrics()
            stats.update(trajectory_stats)
            
            # Confidence statistics
            confidences = [shot.detection_confidence for shot in self.completed_shots]
            if confidences:
                stats['avg_confidence'] = sum(confidences) / len(confidences)
                stats['min_confidence'] = min(confidences)
                stats['max_confidence'] = max(confidences)
                
            return stats
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {}
            
    def reset_session(self):
        """Reset all shot detection state for new session"""
        try:
            self.current_shot = None
            self.completed_shots.clear()
            self.shot_counter = 0
            
            self.ball_detector.reset_tracking()
            self.pose_estimator.reset_history()
            self.trajectory_analyzer.reset()
            
            logger.info("Shot detection session reset")
            
        except Exception as e:
            logger.error(f"Session reset failed: {e}")
            
    def save_shot_data(self, filepath: str) -> bool:
        """Save shot data to file for analysis"""
        try:
            import json
            
            # Convert shot data to serializable format
            shot_data = []
            for shot in self.completed_shots:
                shot_dict = {
                    'shot_id': shot.shot_id,
                    'start_time': float(shot.start_time),
                    'end_time': float(shot.end_time) if shot.end_time is not None else None,
                    'phase': shot.phase.value,
                    'shot_successful': bool(shot.shot_successful) if shot.shot_successful is not None else None,
                    'detection_confidence': float(shot.detection_confidence),
                    'num_poses': len(shot.shooter_poses),
                    'num_ball_positions': len(shot.ball_positions)
                }
                
                # Convert form analysis to dict if available
                if shot.form_analysis:
                    shot_dict['form_analysis'] = {
                        'overall_score': float(shot.form_analysis.overall_score),
                        'elbow_alignment': float(shot.form_analysis.elbow_alignment),
                        'follow_through': float(shot.form_analysis.follow_through),
                        'balance': float(shot.form_analysis.balance),
                        'arc_consistency': float(shot.form_analysis.arc_consistency),
                        'release_timing': float(shot.form_analysis.release_timing),
                        'confidence': float(shot.form_analysis.confidence)
                    }
                
                # Add trajectory data if available
                if shot.trajectory:
                    shot_dict['trajectory'] = {
                        'arc_angle': float(shot.trajectory.arc_angle),
                        'entry_angle': float(shot.trajectory.entry_angle),
                        'shot_distance': float(shot.trajectory.shot_distance),
                        'flight_time': float(shot.trajectory.flight_time),
                        'initial_velocity': float(shot.trajectory.initial_velocity),
                        'confidence': float(shot.trajectory.confidence)
                    }
                    
                shot_data.append(shot_dict)
                
            # Save to file
            with open(filepath, 'w') as f:
                json.dump({
                    'session_stats': self.get_shooting_statistics(),
                    'shots': shot_data
                }, f, indent=2)
                
            logger.info(f"Shot data saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save shot data: {e}")
            return False 