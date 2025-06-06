"""
Biomechanical Analysis Engine

Comprehensive analysis of shooting form and biomechanics including:
- Joint angle calculations for key shooting positions
- Shooting pocket detection and analysis
- Foot positioning and balance analysis
- Follow-through consistency measurement
- Professional coaching standards comparison
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import math
from enum import Enum

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from computer_vision.pose_estimation import ShootingPose, Joint, JointType


class ShootingPhase(Enum):
    """Different phases of the shooting motion"""
    SETUP = "setup"
    LOAD = "load"
    RELEASE = "release"
    FOLLOW_THROUGH = "follow_through"


@dataclass
class JointAngles:
    """Joint angles during shooting motion"""
    # Arm angles (degrees)
    shoulder_angle: float = 0.0      # Shoulder abduction
    elbow_angle: float = 0.0         # Elbow flexion
    wrist_angle: float = 0.0         # Wrist extension
    
    # Leg angles (degrees)
    knee_angle: float = 0.0          # Knee flexion
    ankle_angle: float = 0.0         # Ankle angle
    hip_angle: float = 0.0           # Hip flexion
    
    # Body alignment angles
    spine_angle: float = 0.0         # Spine tilt
    shoulder_alignment: float = 0.0   # Shoulder level difference
    
    # Shooting specific angles
    shooting_pocket_angle: float = 0.0  # Ball position relative to body
    release_angle: float = 0.0          # Release point trajectory angle


@dataclass
class BalanceMetrics:
    """Balance and stability metrics"""
    center_of_mass_x: float = 0.0    # Horizontal COM position
    center_of_mass_y: float = 0.0    # Vertical COM position
    base_of_support_width: float = 0.0  # Foot separation distance
    weight_distribution: float = 0.5    # Left-right weight ratio (0.5 = equal)
    stability_score: float = 0.0        # Overall stability (0-100)
    foot_alignment_score: float = 0.0   # Foot positioning score (0-100)


@dataclass
class FormScore:
    """Comprehensive shooting form scoring"""
    # Individual component scores (0-100)
    elbow_alignment_score: float = 0.0
    shooting_pocket_score: float = 0.0
    balance_score: float = 0.0
    follow_through_score: float = 0.0
    arc_consistency_score: float = 0.0
    release_timing_score: float = 0.0
    
    # Overall weighted score
    overall_score: float = 0.0
    
    # Detailed feedback
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    # Professional comparison
    similarity_to_professional: float = 0.0  # 0-100 similarity score


class BiomechanicalAnalyzer:
    """
    Advanced biomechanical analysis engine for basketball shooting
    
    Analyzes pose data to provide comprehensive form scoring and coaching feedback
    based on professional basketball biomechanics research and coaching principles.
    """
    
    def __init__(self, 
                 scoring_weights: Optional[Dict[str, float]] = None,
                 professional_standards: Optional[Dict[str, Any]] = None):
        """
        Initialize biomechanical analyzer
        
        Args:
            scoring_weights: Custom weights for different form components
            professional_standards: Professional player benchmarks
        """
        # Form component weights (must sum to 1.0)
        self.scoring_weights = scoring_weights or {
            "elbow_alignment": 0.25,
            "shooting_pocket": 0.15,
            "balance": 0.20,
            "follow_through": 0.20,
            "arc_consistency": 0.10,
            "release_timing": 0.10
        }
        
        # Professional shooting standards based on biomechanical research
        self.professional_standards = professional_standards or {
            "elbow_angle_range": (85, 105),      # Degrees at release
            "release_angle_range": (48, 52),     # Arc angle at release
            "shooting_pocket_height": (0.8, 1.0), # Relative to shoulder height
            "balance_tolerance": 0.15,           # Maximum COM deviation
            "follow_through_angle": (15, 35),    # Wrist snap angle
            "knee_flexion_range": (10, 25),      # Degrees during setup
            "foot_width_ratio": (0.8, 1.2)      # Shoulder width multiplier
        }
        
        # Analysis history for trend tracking
        self.form_history: List[FormScore] = []
        
    def analyze_shooting_form(self, 
                             poses: List[ShootingPose],
                             trajectory_data: Optional[Dict[str, Any]] = None) -> FormScore:
        """
        Comprehensive shooting form analysis
        
        Args:
            poses: List of poses throughout shooting motion
            trajectory_data: Ball trajectory information
            
        Returns:
            Detailed form scoring and feedback
        """
        if not poses:
            return FormScore()
        
        # Identify shooting phases
        phase_poses = self._identify_shooting_phases(poses)
        
        # Calculate joint angles for each phase
        joint_angles = self._calculate_joint_angles(phase_poses)
        
        # Analyze balance and stability
        balance_metrics = self._analyze_balance(phase_poses)
        
        # Score individual components
        form_score = FormScore()
        
        # Elbow alignment analysis
        form_score.elbow_alignment_score = self._score_elbow_alignment(joint_angles, phase_poses)
        
        # Shooting pocket analysis
        form_score.shooting_pocket_score = self._score_shooting_pocket(joint_angles, phase_poses)
        
        # Balance analysis
        form_score.balance_score = self._score_balance(balance_metrics)
        
        # Follow-through analysis
        form_score.follow_through_score = self._score_follow_through(joint_angles, phase_poses)
        
        # Arc consistency (requires trajectory data)
        if trajectory_data:
            form_score.arc_consistency_score = self._score_arc_consistency(trajectory_data)
        
        # Release timing
        form_score.release_timing_score = self._score_release_timing(phase_poses)
        
        # Calculate overall weighted score
        form_score.overall_score = self._calculate_overall_score(form_score)
        
        # Generate feedback
        form_score.strengths, form_score.weaknesses, form_score.improvement_suggestions = \
            self._generate_feedback(form_score, joint_angles, balance_metrics)
        
        # Professional comparison
        form_score.similarity_to_professional = self._compare_to_professional_standards(
            form_score, joint_angles, balance_metrics
        )
        
        # Store for trend analysis
        self.form_history.append(form_score)
        
        return form_score
    
    def _identify_shooting_phases(self, poses: List[ShootingPose]) -> Dict[ShootingPhase, List[ShootingPose]]:
        """Identify different phases of the shooting motion"""
        if not poses:
            return {}
        
        # Simple phase identification based on ball height and body position
        # In production, this would use more sophisticated motion analysis
        phase_poses = {phase: [] for phase in ShootingPhase}
        
        # For now, distribute poses across phases based on sequence
        total_poses = len(poses)
        if total_poses >= 4:
            # Divide into phases
            setup_end = total_poses // 4
            load_end = total_poses // 2
            release_end = int(total_poses * 0.75)
            
            phase_poses[ShootingPhase.SETUP] = poses[:setup_end]
            phase_poses[ShootingPhase.LOAD] = poses[setup_end:load_end]
            phase_poses[ShootingPhase.RELEASE] = poses[load_end:release_end]
            phase_poses[ShootingPhase.FOLLOW_THROUGH] = poses[release_end:]
        else:
            # Too few poses, assign to release phase
            phase_poses[ShootingPhase.RELEASE] = poses
        
        return phase_poses
    
    def _calculate_joint_angles(self, phase_poses: Dict[ShootingPhase, List[ShootingPose]]) -> Dict[ShootingPhase, JointAngles]:
        """Calculate joint angles for each shooting phase"""
        joint_angles = {}
        
        for phase, poses in phase_poses.items():
            if not poses:
                joint_angles[phase] = JointAngles()
                continue
                
            # Use the middle pose of each phase for angle calculation
            representative_pose = poses[len(poses) // 2]
            angles = JointAngles()
            
            if representative_pose.joints:
                # Calculate elbow angle (shooting arm)
                angles.elbow_angle = self._calculate_elbow_angle(representative_pose)
                
                # Calculate shoulder angle
                angles.shoulder_angle = self._calculate_shoulder_angle(representative_pose)
                
                # Calculate knee angle
                angles.knee_angle = self._calculate_knee_angle(representative_pose)
                
                # Calculate spine alignment
                angles.spine_angle = self._calculate_spine_angle(representative_pose)
                
                # Calculate shooting pocket angle
                angles.shooting_pocket_angle = self._calculate_shooting_pocket_angle(representative_pose)
                
            joint_angles[phase] = angles
        
        return joint_angles
    
    def _calculate_elbow_angle(self, pose: ShootingPose) -> float:
        """Calculate elbow angle of shooting arm"""
        if not pose.joints:
            return 0.0
        
        # Use dominant hand to determine shooting arm
        if pose.dominant_hand == "right":
            shoulder_type = JointType.RIGHT_SHOULDER
            elbow_type = JointType.RIGHT_ELBOW
            wrist_type = JointType.RIGHT_WRIST
        else:
            shoulder_type = JointType.LEFT_SHOULDER
            elbow_type = JointType.LEFT_ELBOW
            wrist_type = JointType.LEFT_WRIST
        
        if not all(jt in pose.joints for jt in [shoulder_type, elbow_type, wrist_type]):
            return 0.0
        
        shoulder = pose.joints[shoulder_type]
        elbow = pose.joints[elbow_type]
        wrist = pose.joints[wrist_type]
        
        # Calculate angle between shoulder-elbow and elbow-wrist vectors
        vec1 = (shoulder.x - elbow.x, shoulder.y - elbow.y)
        vec2 = (wrist.x - elbow.x, wrist.y - elbow.y)
        
        # Calculate angle using dot product
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
        mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def _calculate_shoulder_angle(self, pose: ShootingPose) -> float:
        """Calculate shoulder abduction angle"""
        if not pose.joints:
            return 0.0
        
        # Calculate angle between shoulders and torso
        if not all(jt in pose.joints for jt in [JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER]):
            return 0.0
        
        left_shoulder = pose.joints[JointType.LEFT_SHOULDER]
        right_shoulder = pose.joints[JointType.RIGHT_SHOULDER]
        
        # Calculate shoulder line angle relative to horizontal
        shoulder_vector = (right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y)
        horizontal_vector = (1, 0)
        
        dot_product = shoulder_vector[0] * horizontal_vector[0] + shoulder_vector[1] * horizontal_vector[1]
        mag_shoulder = math.sqrt(shoulder_vector[0]**2 + shoulder_vector[1]**2)
        
        if mag_shoulder == 0:
            return 0.0
        
        cos_angle = dot_product / mag_shoulder
        cos_angle = max(-1, min(1, cos_angle))
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def _calculate_knee_angle(self, pose: ShootingPose) -> float:
        """Calculate knee flexion angle"""
        if not pose.joints:
            return 0.0
        
        # Use shooting side leg (same as dominant hand)
        if pose.dominant_hand == "right":
            hip_type = JointType.RIGHT_HIP
            knee_type = JointType.RIGHT_KNEE
            ankle_type = JointType.RIGHT_ANKLE
        else:
            hip_type = JointType.LEFT_HIP
            knee_type = JointType.LEFT_KNEE
            ankle_type = JointType.LEFT_ANKLE
        
        if not all(jt in pose.joints for jt in [hip_type, knee_type, ankle_type]):
            return 0.0
        
        hip = pose.joints[hip_type]
        knee = pose.joints[knee_type]
        ankle = pose.joints[ankle_type]
        
        # Calculate angle between hip-knee and knee-ankle vectors
        vec1 = (hip.x - knee.x, hip.y - knee.y)
        vec2 = (ankle.x - knee.x, ankle.y - knee.y)
        
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
        mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))
        angle = math.degrees(math.acos(cos_angle))
        
        # Convert to flexion angle (180 - calculated angle)
        flexion_angle = 180 - angle
        
        return flexion_angle
    
    def _calculate_spine_angle(self, pose: ShootingPose) -> float:
        """Calculate spine alignment angle"""
        if not pose.joints:
            return 0.0
        
        # Calculate spine angle using shoulder midpoint and hip midpoint
        required_joints = [JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER, JointType.LEFT_HIP, JointType.RIGHT_HIP]
        if not all(jt in pose.joints for jt in required_joints):
            return 0.0
        
        left_shoulder = pose.joints[JointType.LEFT_SHOULDER]
        right_shoulder = pose.joints[JointType.RIGHT_SHOULDER]
        left_hip = pose.joints[JointType.LEFT_HIP]
        right_hip = pose.joints[JointType.RIGHT_HIP]
        
        # Calculate midpoints
        shoulder_mid = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
        hip_mid = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
        
        # Calculate spine vector
        spine_vector = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
        vertical_vector = (0, -1)  # Negative Y is up in image coordinates
        
        dot_product = spine_vector[0] * vertical_vector[0] + spine_vector[1] * vertical_vector[1]
        mag_spine = math.sqrt(spine_vector[0]**2 + spine_vector[1]**2)
        
        if mag_spine == 0:
            return 0.0
        
        cos_angle = dot_product / mag_spine
        cos_angle = max(-1, min(1, cos_angle))
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def _calculate_shooting_pocket_angle(self, pose: ShootingPose) -> float:
        """Calculate shooting pocket positioning angle"""
        # This would require ball position relative to body
        # For now, return a placeholder
        return 0.0
    
    def _analyze_balance(self, phase_poses: Dict[ShootingPhase, List[ShootingPose]]) -> BalanceMetrics:
        """Analyze balance and stability throughout shooting motion"""
        balance = BalanceMetrics()
        
        # Use setup and release phases for balance analysis
        setup_poses = phase_poses.get(ShootingPhase.SETUP, [])
        release_poses = phase_poses.get(ShootingPhase.RELEASE, [])
        
        all_poses = setup_poses + release_poses
        if not all_poses:
            return balance
        
        # Calculate average center of mass
        com_x_values = []
        com_y_values = []
        foot_widths = []
        
        for pose in all_poses:
            if pose.joints:
                # Approximate center of mass as midpoint between hips
                if JointType.LEFT_HIP in pose.joints and JointType.RIGHT_HIP in pose.joints:
                    left_hip = pose.joints[JointType.LEFT_HIP]
                    right_hip = pose.joints[JointType.RIGHT_HIP]
                    
                    com_x = (left_hip.x + right_hip.x) / 2
                    com_y = (left_hip.y + right_hip.y) / 2
                    com_x_values.append(com_x)
                    com_y_values.append(com_y)
                
                # Calculate foot separation
                if JointType.LEFT_ANKLE in pose.joints and JointType.RIGHT_ANKLE in pose.joints:
                    left_ankle = pose.joints[JointType.LEFT_ANKLE]
                    right_ankle = pose.joints[JointType.RIGHT_ANKLE]
                    
                    foot_width = abs(right_ankle.x - left_ankle.x)
                    foot_widths.append(foot_width)
        
        if com_x_values:
            balance.center_of_mass_x = sum(com_x_values) / len(com_x_values)
            balance.center_of_mass_y = sum(com_y_values) / len(com_y_values)
            
            # Calculate stability as inverse of COM variance
            com_variance = np.var(com_x_values) + np.var(com_y_values)
            balance.stability_score = max(0, 100 - com_variance / 10)  # Normalized score
        
        if foot_widths:
            balance.base_of_support_width = sum(foot_widths) / len(foot_widths)
            
            # Score foot alignment based on width consistency
            foot_variance = np.var(foot_widths)
            balance.foot_alignment_score = max(0, 100 - foot_variance / 5)
        
        return balance
    
    def _score_elbow_alignment(self, joint_angles: Dict[ShootingPhase, JointAngles], 
                              phase_poses: Dict[ShootingPhase, List[ShootingPose]]) -> float:
        """Score elbow alignment throughout shooting motion"""
        release_angles = joint_angles.get(ShootingPhase.RELEASE)
        if not release_angles:
            return 0.0
        
        elbow_angle = release_angles.elbow_angle
        optimal_range = self.professional_standards["elbow_angle_range"]
        
        # Score based on how close to optimal range
        if optimal_range[0] <= elbow_angle <= optimal_range[1]:
            score = 100.0  # Perfect score within range
        else:
            # Penalty based on distance from range
            if elbow_angle < optimal_range[0]:
                deviation = optimal_range[0] - elbow_angle
            else:
                deviation = elbow_angle - optimal_range[1]
            
            # 5-point penalty per degree of deviation
            score = max(0, 100 - (deviation * 5))
        
        return score
    
    def _score_shooting_pocket(self, joint_angles: Dict[ShootingPhase, JointAngles], 
                              phase_poses: Dict[ShootingPhase, List[ShootingPose]]) -> float:
        """Score shooting pocket positioning"""
        # Placeholder implementation
        # Would analyze ball position relative to shooting shoulder
        return 75.0  # Default reasonable score
    
    def _score_balance(self, balance_metrics: BalanceMetrics) -> float:
        """Score balance and stability"""
        # Combine stability and foot alignment scores
        overall_balance = (balance_metrics.stability_score + balance_metrics.foot_alignment_score) / 2
        return overall_balance
    
    def _score_follow_through(self, joint_angles: Dict[ShootingPhase, JointAngles], 
                             phase_poses: Dict[ShootingPhase, List[ShootingPose]]) -> float:
        """Score follow-through mechanics"""
        follow_through_poses = phase_poses.get(ShootingPhase.FOLLOW_THROUGH, [])
        if not follow_through_poses:
            return 0.0
        
        # Analyze wrist position and arm extension in follow-through
        # This is a simplified scoring - production would analyze full motion
        return 80.0  # Default score
    
    def _score_arc_consistency(self, trajectory_data: Dict[str, Any]) -> float:
        """Score arc consistency based on trajectory data"""
        arc_angle = trajectory_data.get('arc_angle', 0)
        optimal_range = self.professional_standards["release_angle_range"]
        
        if optimal_range[0] <= arc_angle <= optimal_range[1]:
            return 100.0
        else:
            deviation = min(abs(arc_angle - optimal_range[0]), abs(arc_angle - optimal_range[1]))
            return max(0, 100 - (deviation * 10))
    
    def _score_release_timing(self, phase_poses: Dict[ShootingPhase, List[ShootingPose]]) -> float:
        """Score release timing consistency"""
        # Analyze timing consistency across shooting motion
        # Placeholder implementation
        return 85.0  # Default score
    
    def _calculate_overall_score(self, form_score: FormScore) -> float:
        """Calculate weighted overall form score"""
        components = {
            "elbow_alignment": form_score.elbow_alignment_score,
            "shooting_pocket": form_score.shooting_pocket_score,
            "balance": form_score.balance_score,
            "follow_through": form_score.follow_through_score,
            "arc_consistency": form_score.arc_consistency_score,
            "release_timing": form_score.release_timing_score
        }
        
        weighted_score = sum(
            components[component] * weight 
            for component, weight in self.scoring_weights.items()
            if component in components
        )
        
        return weighted_score
    
    def _generate_feedback(self, form_score: FormScore, 
                          joint_angles: Dict[ShootingPhase, JointAngles],
                          balance_metrics: BalanceMetrics) -> Tuple[List[str], List[str], List[str]]:
        """Generate coaching feedback based on analysis"""
        strengths = []
        weaknesses = []
        improvements = []
        
        # Analyze each component
        if form_score.elbow_alignment_score >= 85:
            strengths.append("Excellent elbow alignment throughout shot")
        elif form_score.elbow_alignment_score < 60:
            weaknesses.append("Elbow alignment needs improvement")
            improvements.append("Keep your elbow under the ball and aligned with the basket")
        
        if form_score.balance_score >= 85:
            strengths.append("Great balance and stability")
        elif form_score.balance_score < 60:
            weaknesses.append("Balance issues affecting shot consistency")
            improvements.append("Focus on a stable, shoulder-width stance with even weight distribution")
        
        if form_score.follow_through_score >= 85:
            strengths.append("Consistent follow-through mechanics")
        elif form_score.follow_through_score < 60:
            weaknesses.append("Follow-through needs work")
            improvements.append("Snap your wrist downward and hold your follow-through until the ball hits the rim")
        
        if form_score.arc_consistency_score >= 85:
            strengths.append("Consistent shooting arc")
        elif form_score.arc_consistency_score < 60:
            weaknesses.append("Arc angle inconsistency")
            improvements.append("Aim for a 45-50 degree arc for optimal entry angle")
        
        return strengths, weaknesses, improvements
    
    def _compare_to_professional_standards(self, form_score: FormScore,
                                          joint_angles: Dict[ShootingPhase, JointAngles],
                                          balance_metrics: BalanceMetrics) -> float:
        """Compare shooting form to professional player standards"""
        # Calculate similarity score based on multiple factors
        similarity_factors = []
        
        # Elbow angle similarity
        release_angles = joint_angles.get(ShootingPhase.RELEASE)
        if release_angles:
            elbow_angle = release_angles.elbow_angle
            optimal_range = self.professional_standards["elbow_angle_range"]
            elbow_similarity = 100 if optimal_range[0] <= elbow_angle <= optimal_range[1] else \
                              max(0, 100 - abs(elbow_angle - np.mean(optimal_range)) * 5)
            similarity_factors.append(elbow_similarity)
        
        # Balance similarity
        balance_similarity = balance_metrics.stability_score
        similarity_factors.append(balance_similarity)
        
        # Overall form score contribution
        form_similarity = form_score.overall_score
        similarity_factors.append(form_similarity)
        
        # Calculate weighted average
        if similarity_factors:
            return sum(similarity_factors) / len(similarity_factors)
        
        return 0.0
    
    def get_improvement_trends(self, sessions: int = 10) -> Dict[str, Any]:
        """Analyze improvement trends over recent sessions"""
        if len(self.form_history) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        recent_scores = self.form_history[-sessions:]
        
        # Calculate trends for each component
        trends = {}
        components = ["elbow_alignment_score", "balance_score", "follow_through_score", 
                     "arc_consistency_score", "overall_score"]
        
        for component in components:
            scores = [getattr(score, component) for score in recent_scores]
            if len(scores) >= 2:
                # Simple linear trend calculation
                x = list(range(len(scores)))
                slope = np.polyfit(x, scores, 1)[0]
                trends[component] = {
                    "trend": "improving" if slope > 1 else "declining" if slope < -1 else "stable",
                    "rate": slope,
                    "current": scores[-1],
                    "sessions_analyzed": len(scores)
                }
        
        return trends 