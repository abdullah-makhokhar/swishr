"""
Pose Estimation Module for Basketball Shooting Analysis
Uses MediaPipe for real-time human pose detection and analysis
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict, NamedTuple
import logging
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class JointType(Enum):
    """Basketball-relevant joint types"""
    # Upper body shooting form joints
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    NOSE = "nose"


@dataclass
class Joint:
    """Joint position and properties"""
    x: float
    y: float
    z: float  # Depth from MediaPipe
    visibility: float
    joint_type: JointType
    timestamp: float


@dataclass
class ShootingPose:
    """Complete pose data for shooting analysis"""
    joints: Dict[JointType, Joint]
    timestamp: float
    confidence: float
    is_shooting_stance: bool = False
    dominant_hand: Optional[str] = None  # "left" or "right"


class PoseEstimator:
    """
    Pose estimation for basketball shooting form analysis
    
    Uses MediaPipe Pose to track key joints and analyze shooting mechanics
    including release point, follow-through, and body alignment.
    """
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize pose estimator
        
        Args:
            static_image_mode: Whether to process static images or video stream
            model_complexity: Complexity of pose model (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks across frames
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # MediaPipe landmark indices mapping to our joint types
        self.landmark_mapping = {
            JointType.NOSE: 0,
            JointType.LEFT_SHOULDER: 11,
            JointType.RIGHT_SHOULDER: 12,
            JointType.LEFT_ELBOW: 13,
            JointType.RIGHT_ELBOW: 14,
            JointType.LEFT_WRIST: 15,
            JointType.RIGHT_WRIST: 16,
            JointType.LEFT_HIP: 23,
            JointType.RIGHT_HIP: 24,
            JointType.LEFT_KNEE: 25,
            JointType.RIGHT_KNEE: 26,
            JointType.LEFT_ANKLE: 27,
            JointType.RIGHT_ANKLE: 28,
        }
        
        # Pose history for temporal analysis
        self.pose_history: List[ShootingPose] = []
        self.max_history_size = 30  # Keep last 30 poses (1 second at 30fps)
        
    def detect_pose(self, frame: np.ndarray) -> Optional[ShootingPose]:
        """
        Detect pose in frame and return shooting pose data
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            ShootingPose object or None if no pose detected
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return None
                
            # Extract joint positions
            joints = {}
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            
            for joint_type, landmark_idx in self.landmark_mapping.items():
                landmark = results.pose_landmarks.landmark[landmark_idx]
                
                # Convert normalized coordinates to pixel coordinates
                h, w = frame.shape[:2]
                
                joint = Joint(
                    x=landmark.x * w,
                    y=landmark.y * h,
                    z=landmark.z,  # Relative depth
                    visibility=landmark.visibility,
                    joint_type=joint_type,
                    timestamp=current_time
                )
                joints[joint_type] = joint
                
            # Create shooting pose
            shooting_pose = ShootingPose(
                joints=joints,
                timestamp=current_time,
                confidence=self._calculate_pose_confidence(joints),
                is_shooting_stance=self._is_shooting_stance(joints),
                dominant_hand=self._determine_dominant_hand(joints)
            )
            
            # Add to history
            self.pose_history.append(shooting_pose)
            if len(self.pose_history) > self.max_history_size:
                self.pose_history.pop(0)
                
            return shooting_pose
            
        except Exception as e:
            logger.error(f"Pose detection failed: {e}")
            return None
            
    def _calculate_pose_confidence(self, joints: Dict[JointType, Joint]) -> float:
        """Calculate overall confidence of pose detection"""
        if not joints:
            return 0.0
            
        # Average visibility of key shooting joints
        key_joints = [
            JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER,
            JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW,
            JointType.LEFT_WRIST, JointType.RIGHT_WRIST
        ]
        
        visible_joints = [joints[jt] for jt in key_joints if jt in joints]
        if not visible_joints:
            return 0.0
            
        avg_visibility = sum(joint.visibility for joint in visible_joints) / len(visible_joints)
        return avg_visibility
        
    def _is_shooting_stance(self, joints: Dict[JointType, Joint]) -> bool:
        """
        Determine if current pose represents a shooting stance
        
        Analyzes body position and joint angles to identify shooting motion
        """
        try:
            # Check if key joints are visible
            required_joints = [
                JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER,
                JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW,
                JointType.LEFT_WRIST, JointType.RIGHT_WRIST
            ]
            
            if not all(jt in joints for jt in required_joints):
                return False
                
            # Check for raised arms (shooting position)
            left_shoulder = joints[JointType.LEFT_SHOULDER]
            right_shoulder = joints[JointType.RIGHT_SHOULDER]
            left_elbow = joints[JointType.LEFT_ELBOW]
            right_elbow = joints[JointType.RIGHT_ELBOW]
            left_wrist = joints[JointType.LEFT_WRIST]
            right_wrist = joints[JointType.RIGHT_WRIST]
            
            # Calculate arm elevation angles
            left_arm_angle = self._calculate_arm_elevation(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = self._calculate_arm_elevation(right_shoulder, right_elbow, right_wrist)
            
            # Shooting stance typically has at least one arm raised significantly
            min_shooting_angle = 30  # degrees
            max_shooting_angle = 150  # degrees
            
            left_shooting = min_shooting_angle < left_arm_angle < max_shooting_angle
            right_shooting = min_shooting_angle < right_arm_angle < max_shooting_angle
            
            return left_shooting or right_shooting
            
        except Exception as e:
            logger.error(f"Shooting stance detection failed: {e}")
            return False
            
    def _determine_dominant_hand(self, joints: Dict[JointType, Joint]) -> Optional[str]:
        """Determine which hand is dominant based on arm position"""
        try:
            if not all(jt in joints for jt in [JointType.LEFT_WRIST, JointType.RIGHT_WRIST,
                                             JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER]):
                return None
                
            left_wrist = joints[JointType.LEFT_WRIST]
            right_wrist = joints[JointType.RIGHT_WRIST]
            left_shoulder = joints[JointType.LEFT_SHOULDER]
            right_shoulder = joints[JointType.RIGHT_SHOULDER]
            
            # Calculate wrist height relative to shoulders
            left_wrist_elevation = left_shoulder.y - left_wrist.y
            right_wrist_elevation = right_shoulder.y - right_wrist.y
            
            # Higher wrist typically indicates shooting hand
            if left_wrist_elevation > right_wrist_elevation + 20:  # 20 pixel threshold
                return "left"
            elif right_wrist_elevation > left_wrist_elevation + 20:
                return "right"
            else:
                return None  # Ambiguous
                
        except Exception as e:
            logger.error(f"Could not determine dominant hand: {e}")
            return None
            
    def _calculate_arm_elevation(self, shoulder: Joint, elbow: Joint, wrist: Joint) -> float:
        """Calculate arm elevation angle in degrees"""
        try:
            # Vector from shoulder to elbow
            shoulder_to_elbow = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y])
            
            # Vector from elbow to wrist  
            elbow_to_wrist = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
            
            # Calculate angle at elbow
            dot_product = np.dot(shoulder_to_elbow, elbow_to_wrist)
            norms = np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_wrist)
            
            if norms == 0:
                return 0.0
                
            cos_angle = dot_product / norms
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
            
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            logger.error(f"Arm elevation calculation failed: {e}")
            return 0.0
            
    def analyze_shooting_form(self, pose: ShootingPose) -> Dict[str, float]:
        """
        Analyze shooting form metrics from pose data
        
        Args:
            pose: ShootingPose to analyze
            
        Returns:
            Dictionary of shooting form metrics and scores
        """
        metrics = {}
        
        try:
            if not pose.is_shooting_stance:
                return metrics
                
            joints = pose.joints
            
            # 1. Elbow alignment analysis
            if pose.dominant_hand:
                elbow_alignment = self._analyze_elbow_alignment(joints, pose.dominant_hand)
                metrics["elbow_alignment_score"] = elbow_alignment
                
            # 2. Shooting arc analysis (requires trajectory data)
            if len(self.pose_history) >= 5:
                arc_consistency = self._analyze_arc_consistency()
                metrics["arc_consistency_score"] = arc_consistency
                
            # 3. Follow-through analysis
            follow_through = self._analyze_follow_through(joints, pose.dominant_hand)
            metrics["follow_through_score"] = follow_through
            
            # 4. Base and balance analysis
            balance_score = self._analyze_balance(joints)
            metrics["balance_score"] = balance_score
            
            # 5. Overall form score (weighted average)
            weights = {
                "elbow_alignment_score": 0.25,
                "arc_consistency_score": 0.20,
                "follow_through_score": 0.25,
                "balance_score": 0.30
            }
            
            overall_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    overall_score += metrics[metric] * weight
                    total_weight += weight
                    
            if total_weight > 0:
                metrics["overall_form_score"] = overall_score / total_weight
                
        except Exception as e:
            logger.error(f"Shooting form analysis failed: {e}")
            
        return metrics
        
    def _analyze_elbow_alignment(self, joints: Dict[JointType, Joint], dominant_hand: str) -> float:
        """Analyze elbow alignment for shooting form"""
        try:
            if dominant_hand == "right":
                shoulder = joints.get(JointType.RIGHT_SHOULDER)
                elbow = joints.get(JointType.RIGHT_ELBOW)
                wrist = joints.get(JointType.RIGHT_WRIST)
            else:
                shoulder = joints.get(JointType.LEFT_SHOULDER)
                elbow = joints.get(JointType.LEFT_ELBOW)
                wrist = joints.get(JointType.LEFT_WRIST)
                
            if not all([shoulder, elbow, wrist]):
                return 0.0
                
            # Ideal elbow position is directly under the wrist and in line with shoulder
            # Calculate horizontal alignment
            shoulder_elbow_distance = abs(elbow.x - shoulder.x)
            elbow_wrist_distance = abs(wrist.x - elbow.x)
            
            # Ideal alignment score (closer to 0 is better)
            alignment_error = abs(shoulder_elbow_distance - elbow_wrist_distance)
            
            # Convert to score (0-100, higher is better)
            max_error = 50  # pixels
            alignment_score = max(0, 100 - (alignment_error / max_error) * 100)
            
            return alignment_score
            
        except Exception as e:
            logger.error(f"Elbow alignment analysis failed: {e}")
            return 0.0
            
    def _analyze_arc_consistency(self) -> float:
        """Analyze consistency of shooting arc across recent poses"""
        try:
            if len(self.pose_history) < 5:
                return 0.0
                
            # Get wrist heights from recent poses during shooting motion
            wrist_heights = []
            
            for pose in self.pose_history[-10:]:  # Last 10 poses
                if pose.is_shooting_stance and pose.dominant_hand:
                    wrist_joint = (JointType.RIGHT_WRIST if pose.dominant_hand == "right" 
                                 else JointType.LEFT_WRIST)
                    
                    if wrist_joint in pose.joints:
                        wrist_heights.append(pose.joints[wrist_joint].y)
                        
            if len(wrist_heights) < 3:
                return 0.0
                
            # Calculate consistency (lower standard deviation = higher consistency)
            height_std = np.std(wrist_heights)
            
            # Convert to score (0-100)
            max_std = 30  # pixels
            consistency_score = max(0, 100 - (height_std / max_std) * 100)
            
            return consistency_score
            
        except Exception as e:
            logger.error(f"Arc consistency analysis failed: {e}")
            return 0.0
            
    def _analyze_follow_through(self, joints: Dict[JointType, Joint], dominant_hand: Optional[str]) -> float:
        """Analyze follow-through mechanics"""
        try:
            if not dominant_hand:
                return 0.0
                
            # Get shooting arm joints
            if dominant_hand == "right":
                elbow = joints.get(JointType.RIGHT_ELBOW)
                wrist = joints.get(JointType.RIGHT_WRIST)
            else:
                elbow = joints.get(JointType.LEFT_ELBOW)
                wrist = joints.get(JointType.LEFT_WRIST)
                
            if not all([elbow, wrist]):
                return 0.0
                
            # Ideal follow-through has wrist below elbow (downward angle)
            wrist_drop = elbow.y - wrist.y
            
            # Positive wrist drop indicates good follow-through
            if wrist_drop > 0:
                # Score based on amount of wrist drop
                max_drop = 40  # pixels
                follow_through_score = min(100, (wrist_drop / max_drop) * 100)
            else:
                # Penalty for upward wrist position
                follow_through_score = max(0, 50 + wrist_drop)  # wrist_drop is negative
                
            return follow_through_score
            
        except Exception as e:
            logger.error(f"Follow-through analysis failed: {e}")
            return 0.0
            
    def _analyze_balance(self, joints: Dict[JointType, Joint]) -> float:
        """Analyze body balance and stance"""
        try:
            # Get key joints for balance analysis
            left_hip = joints.get(JointType.LEFT_HIP)
            right_hip = joints.get(JointType.RIGHT_HIP)
            left_ankle = joints.get(JointType.LEFT_ANKLE)
            right_ankle = joints.get(JointType.RIGHT_ANKLE)
            
            if not all([left_hip, right_hip, left_ankle, right_ankle]):
                return 0.0
                
            # Calculate hip center and foot center
            hip_center_x = (left_hip.x + right_hip.x) / 2
            foot_center_x = (left_ankle.x + right_ankle.x) / 2
            
            # Good balance has hips over feet
            balance_offset = abs(hip_center_x - foot_center_x)
            
            # Calculate stance width (should be shoulder-width apart)
            stance_width = abs(right_ankle.x - left_ankle.x)
            hip_width = abs(right_hip.x - left_hip.x)
            
            # Ideal stance is about hip-width apart
            stance_ratio = min(stance_width, hip_width) / max(stance_width, hip_width)
            
            # Balance score combines center of gravity and stance width
            balance_error_score = max(0, 100 - (balance_offset / 30) * 100)  # 30 pixel tolerance
            stance_score = stance_ratio * 100
            
            balance_score = (balance_error_score + stance_score) / 2
            
            return balance_score
            
        except Exception as e:
            logger.error(f"Balance analysis failed: {e}")
            return 0.0
            
    def get_joint_angles(self, pose: ShootingPose) -> Dict[str, float]:
        """Calculate key joint angles for biomechanical analysis"""
        angles = {}
        
        try:
            joints = pose.joints
            
            # Elbow angles
            if all(jt in joints for jt in [JointType.LEFT_SHOULDER, JointType.LEFT_ELBOW, JointType.LEFT_WRIST]):
                left_elbow_angle = self._calculate_joint_angle(
                    joints[JointType.LEFT_SHOULDER],
                    joints[JointType.LEFT_ELBOW],
                    joints[JointType.LEFT_WRIST]
                )
                angles["left_elbow_angle"] = left_elbow_angle
                
            if all(jt in joints for jt in [JointType.RIGHT_SHOULDER, JointType.RIGHT_ELBOW, JointType.RIGHT_WRIST]):
                right_elbow_angle = self._calculate_joint_angle(
                    joints[JointType.RIGHT_SHOULDER],
                    joints[JointType.RIGHT_ELBOW],
                    joints[JointType.RIGHT_WRIST]
                )
                angles["right_elbow_angle"] = right_elbow_angle
                
            # Knee angles
            if all(jt in joints for jt in [JointType.LEFT_HIP, JointType.LEFT_KNEE, JointType.LEFT_ANKLE]):
                left_knee_angle = self._calculate_joint_angle(
                    joints[JointType.LEFT_HIP],
                    joints[JointType.LEFT_KNEE],
                    joints[JointType.LEFT_ANKLE]
                )
                angles["left_knee_angle"] = left_knee_angle
                
            if all(jt in joints for jt in [JointType.RIGHT_HIP, JointType.RIGHT_KNEE, JointType.RIGHT_ANKLE]):
                right_knee_angle = self._calculate_joint_angle(
                    joints[JointType.RIGHT_HIP],
                    joints[JointType.RIGHT_KNEE],
                    joints[JointType.RIGHT_ANKLE]
                )
                angles["right_knee_angle"] = right_knee_angle
                
        except Exception as e:
            logger.error(f"Joint angle calculation failed: {e}")
            
        return angles
        
    def _calculate_joint_angle(self, joint1: Joint, joint2: Joint, joint3: Joint) -> float:
        """Calculate angle at joint2 formed by joint1-joint2-joint3"""
        try:
            # Vectors from joint2 to joint1 and joint3
            vec1 = np.array([joint1.x - joint2.x, joint1.y - joint2.y])
            vec2 = np.array([joint3.x - joint2.x, joint3.y - joint2.y])
            
            # Calculate angle
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            
            if norms == 0:
                return 0.0
                
            cos_angle = dot_product / norms
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            logger.error(f"Joint angle calculation failed: {e}")
            return 0.0
            
    def draw_pose(self, frame: np.ndarray, pose: ShootingPose) -> np.ndarray:
        """
        Draw pose landmarks and connections on frame
        
        Args:
            frame: Input frame
            pose: ShootingPose to draw
            
        Returns:
            Frame with pose overlay
        """
        try:
            # Simple joint drawing approach to avoid MediaPipe compatibility issues
            # Draw key joints as circles
            for joint_type, joint in pose.joints.items():
                if joint.visibility > 0.5:  # Only draw visible joints
                    center = (int(joint.x), int(joint.y))
                    
                    # Color code different joint types
                    if joint_type in [JointType.LEFT_WRIST, JointType.RIGHT_WRIST]:
                        color = (0, 255, 255)  # Yellow for wrists
                    elif joint_type in [JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW]:
                        color = (255, 0, 255)  # Magenta for elbows
                    elif joint_type in [JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER]:
                        color = (0, 255, 0)    # Green for shoulders
                    else:
                        color = (255, 255, 255)  # White for other joints
                        
                    cv2.circle(frame, center, 5, color, -1)
                    cv2.circle(frame, center, 7, (0, 0, 0), 2)  # Black outline
            
            # Draw connections between key joints
            connections = [
                (JointType.LEFT_SHOULDER, JointType.LEFT_ELBOW),
                (JointType.LEFT_ELBOW, JointType.LEFT_WRIST),
                (JointType.RIGHT_SHOULDER, JointType.RIGHT_ELBOW),
                (JointType.RIGHT_ELBOW, JointType.RIGHT_WRIST),
                (JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER),
                (JointType.LEFT_HIP, JointType.RIGHT_HIP),
                (JointType.LEFT_SHOULDER, JointType.LEFT_HIP),
                (JointType.RIGHT_SHOULDER, JointType.RIGHT_HIP),
            ]
            
            for joint1_type, joint2_type in connections:
                if (joint1_type in pose.joints and joint2_type in pose.joints and
                    pose.joints[joint1_type].visibility > 0.5 and 
                    pose.joints[joint2_type].visibility > 0.5):
                    
                    pt1 = (int(pose.joints[joint1_type].x), int(pose.joints[joint1_type].y))
                    pt2 = (int(pose.joints[joint2_type].x), int(pose.joints[joint2_type].y))
                    cv2.line(frame, pt1, pt2, (100, 100, 100), 2)
            
            # Add shooting form indicators
            if pose.is_shooting_stance:
                cv2.putText(frame, "SHOOTING STANCE", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                           
            if pose.dominant_hand:
                cv2.putText(frame, f"Dominant: {pose.dominant_hand.upper()}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                           
            # Add confidence score
            cv2.putText(frame, f"Pose Confidence: {pose.confidence:.2f}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                           
        except Exception as e:
            logger.error(f"Pose drawing failed: {e}")
            
        return frame
        
    def reset_history(self):
        """Reset pose history"""
        self.pose_history.clear() 