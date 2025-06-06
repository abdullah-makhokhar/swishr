"""
Unit tests for basketball analytics modules
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analytics.performance_metrics import PerformanceAnalyzer, ShootingMetrics, ShotChartPoint
from analytics.biomechanical_analyzer import BiomechanicalAnalyzer, FormScore, JointAngles, BalanceMetrics, ShootingPhase
from computer_vision.shot_detection import ShotEvent, FormAnalysis
from computer_vision.trajectory_analysis import ShotTrajectory, TrajectoryPoint
from computer_vision.pose_estimation import ShootingPose, Joint, JointType


class TestPerformanceAnalyzer:
    """Test cases for PerformanceAnalyzer class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.analyzer = PerformanceAnalyzer()
        self.sample_shots = self._create_sample_shots()
    
    def _create_sample_shots(self) -> list:
        """Create sample shot data for testing"""
        shots = []
        for i in range(10):
            # Create sample trajectory points
            points = [
                TrajectoryPoint(x=100 + j * 10, y=200 - j * 5, timestamp=j * 0.1) 
                for j in range(5)
            ]
            
            trajectory = ShotTrajectory(
                points=points,
                release_point=points[0] if points else None,
                apex_point=points[2] if len(points) > 2 else None,
                landing_point=points[-1] if points else None,
                initial_velocity=15.0,
                arc_angle=47.0 + i,
                entry_angle=42.0,
                peak_height=12.0,
                shot_distance=15.0 + i * 2,
                flight_time=1.5
            )
            
            form_analysis = FormAnalysis(
                overall_score=75.0 + i * 2,
                elbow_alignment=80.0,
                follow_through=70.0,
                balance=85.0,
                arc_consistency=80.0
            )
            
            shot = ShotEvent(
                shot_successful=(i % 3 == 0),  # 33% success rate
                trajectory=trajectory,
                form_analysis=form_analysis,
                detection_confidence=0.85,
                timestamp=datetime.now() - timedelta(minutes=i)
            )
            shots.append(shot)
        
        return shots
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        assert self.analyzer is not None
        assert len(self.analyzer.distance_zones) == 4
        assert "close" in self.analyzer.distance_zones
        assert "three_point" in self.analyzer.distance_zones
        assert len(self.analyzer.form_weights) == 4
        
    def test_session_analysis(self):
        """Test complete session analysis"""
        session_start = datetime.now() - timedelta(hours=1)
        session_end = datetime.now()
        
        metrics = self.analyzer.analyze_session(
            self.sample_shots, session_start, session_end
        )
        
        assert isinstance(metrics, ShootingMetrics)
        assert metrics.total_shots == 10
        assert metrics.made_shots > 0
        assert 0 <= metrics.shooting_percentage <= 100
        assert metrics.session_duration > 0
        assert len(metrics.shot_chart_data) > 0
        
    def test_distance_zone_analysis(self):
        """Test distance-based performance analysis"""
        session_start = datetime.now() - timedelta(hours=1)
        session_end = datetime.now()
        
        metrics = self.analyzer.analyze_session(
            self.sample_shots, session_start, session_end
        )
        
        assert "mid_range" in metrics.shots_by_distance
        assert "three_point" in metrics.shots_by_distance
        
        # Check mid-range stats
        mid_range_stats = metrics.shots_by_distance["mid_range"]
        assert "total_shots" in mid_range_stats
        assert "shooting_percentage" in mid_range_stats
        assert mid_range_stats["total_shots"] >= 0
        
    def test_arc_consistency_calculation(self):
        """Test arc angle consistency calculation"""
        session_start = datetime.now() - timedelta(hours=1)
        session_end = datetime.now()
        
        metrics = self.analyzer.analyze_session(
            self.sample_shots, session_start, session_end
        )
        
        assert metrics.average_arc_angle > 0
        assert metrics.arc_consistency >= 0  # Standard deviation
        
    def test_form_consistency_analysis(self):
        """Test form score consistency analysis"""
        session_start = datetime.now() - timedelta(hours=1)
        session_end = datetime.now()
        
        metrics = self.analyzer.analyze_session(
            self.sample_shots, session_start, session_end
        )
        
        assert metrics.form_score_average > 0
        assert metrics.form_score_consistency >= 0
        
    def test_empty_shots_handling(self):
        """Test handling of empty shot list"""
        session_start = datetime.now() - timedelta(hours=1)
        session_end = datetime.now()
        
        metrics = self.analyzer.analyze_session([], session_start, session_end)
        
        assert metrics.total_shots == 0
        assert metrics.shooting_percentage == 0.0
        assert len(metrics.shot_chart_data) == 0
        
    def test_performance_summary(self):
        """Test performance summary generation"""
        # First add some session history
        session_start = datetime.now() - timedelta(hours=1)
        session_end = datetime.now()
        
        self.analyzer.analyze_session(self.sample_shots, session_start, session_end)
        
        summary = self.analyzer.get_performance_summary(days=30)
        
        assert "total_sessions" in summary
        assert "overall_shooting_percentage" in summary
        assert "best_distance_zone" in summary
        assert summary["total_sessions"] >= 1
        
    def test_metrics_export_json(self):
        """Test JSON export functionality"""
        # Add session data
        session_start = datetime.now() - timedelta(hours=1)
        session_end = datetime.now()
        
        self.analyzer.analyze_session(self.sample_shots, session_start, session_end)
        
        # Test export (without actually writing file)
        success = self.analyzer.export_metrics("test_export.json", "json")
        assert success is True  # Should succeed in creating export data
        
    def test_improvement_rate_calculation(self):
        """Test improvement rate calculation with multiple sessions"""
        # Add multiple sessions
        for i in range(3):
            session_start = datetime.now() - timedelta(days=i+1)
            session_end = session_start + timedelta(hours=1)
            
            # Gradually improve shooting percentage
            improved_shots = []
            for j, shot in enumerate(self.sample_shots):
                improved_shot = shot
                improved_shot.shot_successful = (j < (5 + i))  # Improve over time
                improved_shots.append(improved_shot)
            
            self.analyzer.analyze_session(improved_shots, session_start, session_end)
        
        # Last session should show improvement
        latest_metrics = self.analyzer.session_history[-1]
        assert hasattr(latest_metrics, 'improvement_rate')


class TestBiomechanicalAnalyzer:
    """Test cases for BiomechanicalAnalyzer class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.analyzer = BiomechanicalAnalyzer()
        self.sample_poses = self._create_sample_poses()
        
    def _create_sample_poses(self) -> list:
        """Create sample pose data for testing"""
        poses = []
        
        for i in range(8):  # 8 poses for complete shooting motion
            joints = {
                JointType.NOSE: Joint(320 + i, 100 + i * 2, 0, 0.9, JointType.NOSE, i * 0.1),
                JointType.LEFT_SHOULDER: Joint(280 + i, 150 + i * 3, 0, 0.9, JointType.LEFT_SHOULDER, i * 0.1),
                JointType.RIGHT_SHOULDER: Joint(360 + i, 150 + i * 3, 0, 0.9, JointType.RIGHT_SHOULDER, i * 0.1),
                JointType.LEFT_ELBOW: Joint(250 + i, 200 + i * 4, 0, 0.9, JointType.LEFT_ELBOW, i * 0.1),
                JointType.RIGHT_ELBOW: Joint(390 + i, 200 + i * 4, 0, 0.9, JointType.RIGHT_ELBOW, i * 0.1),
                JointType.LEFT_WRIST: Joint(220 + i, 250 + i * 5, 0, 0.9, JointType.LEFT_WRIST, i * 0.1),
                JointType.RIGHT_WRIST: Joint(420 + i, 250 + i * 5, 0, 0.9, JointType.RIGHT_WRIST, i * 0.1),
                JointType.LEFT_HIP: Joint(300 + i, 350 + i * 2, 0, 0.9, JointType.LEFT_HIP, i * 0.1),
                JointType.RIGHT_HIP: Joint(340 + i, 350 + i * 2, 0, 0.9, JointType.RIGHT_HIP, i * 0.1),
                JointType.LEFT_KNEE: Joint(290 + i, 450 + i, 0, 0.9, JointType.LEFT_KNEE, i * 0.1),
                JointType.RIGHT_KNEE: Joint(350 + i, 450 + i, 0, 0.9, JointType.RIGHT_KNEE, i * 0.1),
                JointType.LEFT_ANKLE: Joint(285 + i, 550, 0, 0.9, JointType.LEFT_ANKLE, i * 0.1),
                JointType.RIGHT_ANKLE: Joint(355 + i, 550, 0, 0.9, JointType.RIGHT_ANKLE, i * 0.1)
            }
            
            pose = ShootingPose(
                joints=joints,
                confidence=0.85,
                timestamp=i * 0.1,
                dominant_hand="right"
            )
            poses.append(pose)
            
        return poses
    
    def test_analyzer_initialization(self):
        """Test biomechanical analyzer initialization"""
        assert self.analyzer is not None
        assert len(self.analyzer.scoring_weights) == 6
        assert sum(self.analyzer.scoring_weights.values()) == pytest.approx(1.0, rel=1e-2)
        assert len(self.analyzer.professional_standards) > 0
        
    def test_shooting_form_analysis(self):
        """Test complete shooting form analysis"""
        trajectory_data = {
            'arc_angle': 49.0,
            'entry_angle': 43.0,
            'shot_distance': 18.0
        }
        
        form_score = self.analyzer.analyze_shooting_form(
            self.sample_poses, trajectory_data
        )
        
        assert isinstance(form_score, FormScore)
        assert 0 <= form_score.overall_score <= 100
        assert 0 <= form_score.elbow_alignment_score <= 100
        assert 0 <= form_score.balance_score <= 100
        assert form_score.similarity_to_professional >= 0
        
    def test_shooting_phases_identification(self):
        """Test shooting phases identification"""
        phase_poses = self.analyzer._identify_shooting_phases(self.sample_poses)
        
        assert ShootingPhase.SETUP in phase_poses
        assert ShootingPhase.RELEASE in phase_poses
        assert ShootingPhase.FOLLOW_THROUGH in phase_poses
        
        # Should have poses in each phase
        total_poses_in_phases = sum(len(poses) for poses in phase_poses.values())
        assert total_poses_in_phases == len(self.sample_poses)
        
    def test_joint_angle_calculations(self):
        """Test joint angle calculations"""
        phase_poses = self.analyzer._identify_shooting_phases(self.sample_poses)
        joint_angles = self.analyzer._calculate_joint_angles(phase_poses)
        
        assert ShootingPhase.RELEASE in joint_angles
        release_angles = joint_angles[ShootingPhase.RELEASE]
        
        assert isinstance(release_angles, JointAngles)
        assert release_angles.elbow_angle >= 0
        assert release_angles.shoulder_angle >= 0
        assert release_angles.knee_angle >= 0
        
    def test_elbow_angle_calculation(self):
        """Test elbow angle calculation for shooting arm"""
        pose = self.sample_poses[4]  # Use middle pose
        elbow_angle = self.analyzer._calculate_elbow_angle(pose)
        
        assert 0 <= elbow_angle <= 180  # Valid angle range
        
    def test_balance_analysis(self):
        """Test balance and stability analysis"""
        phase_poses = self.analyzer._identify_shooting_phases(self.sample_poses)
        balance_metrics = self.analyzer._analyze_balance(phase_poses)
        
        assert isinstance(balance_metrics, BalanceMetrics)
        assert balance_metrics.center_of_mass_x > 0
        assert balance_metrics.center_of_mass_y > 0
        assert 0 <= balance_metrics.stability_score <= 100
        assert 0 <= balance_metrics.foot_alignment_score <= 100
        
    def test_form_scoring_components(self):
        """Test individual form scoring components"""
        phase_poses = self.analyzer._identify_shooting_phases(self.sample_poses)
        joint_angles = self.analyzer._calculate_joint_angles(phase_poses)
        
        # Test elbow alignment scoring
        elbow_score = self.analyzer._score_elbow_alignment(joint_angles, phase_poses)
        assert 0 <= elbow_score <= 100
        
        # Test balance scoring
        balance_metrics = self.analyzer._analyze_balance(phase_poses)
        balance_score = self.analyzer._score_balance(balance_metrics)
        assert 0 <= balance_score <= 100
        
    def test_feedback_generation(self):
        """Test coaching feedback generation"""
        form_score = FormScore(
            elbow_alignment_score=90.0,
            balance_score=60.0,  # Below threshold
            follow_through_score=85.0,
            arc_consistency_score=50.0  # Below threshold
        )
        
        joint_angles = {ShootingPhase.RELEASE: JointAngles()}
        balance_metrics = BalanceMetrics()
        
        strengths, weaknesses, improvements = self.analyzer._generate_feedback(
            form_score, joint_angles, balance_metrics
        )
        
        assert isinstance(strengths, list)
        assert isinstance(weaknesses, list)
        assert isinstance(improvements, list)
        assert len(weaknesses) > 0  # Should identify low scores
        assert len(improvements) > 0  # Should provide suggestions
        
    def test_professional_comparison(self):
        """Test comparison to professional standards"""
        form_score = FormScore(overall_score=85.0)
        joint_angles = {
            ShootingPhase.RELEASE: JointAngles(elbow_angle=95.0)  # Within optimal range
        }
        balance_metrics = BalanceMetrics(stability_score=90.0)
        
        similarity = self.analyzer._compare_to_professional_standards(
            form_score, joint_angles, balance_metrics
        )
        
        assert 0 <= similarity <= 100
        assert similarity > 70  # Should be high with good scores
        
    def test_empty_poses_handling(self):
        """Test handling of empty pose list"""
        form_score = self.analyzer.analyze_shooting_form([])
        
        assert isinstance(form_score, FormScore)
        assert form_score.overall_score == 0.0
        assert len(form_score.strengths) == 0
        
    def test_improvement_trends(self):
        """Test improvement trend analysis"""
        # Add multiple form scores to history
        for i in range(5):
            score = FormScore(
                overall_score=70.0 + i * 5,  # Improving trend
                elbow_alignment_score=75.0 + i * 3,
                balance_score=80.0 + i * 2
            )
            self.analyzer.form_history.append(score)
        
        trends = self.analyzer.get_improvement_trends(sessions=5)
        
        assert "overall_score" in trends
        assert trends["overall_score"]["trend"] == "improving"
        assert trends["overall_score"]["rate"] > 0
        
    def test_insufficient_trend_data(self):
        """Test trend analysis with insufficient data"""
        trends = self.analyzer.get_improvement_trends()
        
        assert "error" in trends
        assert "Insufficient data" in trends["error"]


class TestAnalyticsIntegration:
    """Integration tests for analytics components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.performance_analyzer = PerformanceAnalyzer()
        self.biomech_analyzer = BiomechanicalAnalyzer()
        
    def test_end_to_end_analysis(self):
        """Test complete end-to-end analytics workflow"""
        # Create sample data
        shots = []
        poses = []
        
        # Create coordinated shot and pose data
        for i in range(5):
            # Create sample trajectory points
            points = [
                TrajectoryPoint(x=120 + j * 12, y=180 - j * 6, timestamp=j * 0.12) 
                for j in range(6)
            ]
            
            # Create trajectory
            trajectory = ShotTrajectory(
                points=points,
                release_point=points[0] if points else None,
                apex_point=points[3] if len(points) > 3 else None,
                landing_point=points[-1] if points else None,
                initial_velocity=16.0,
                shot_distance=16.0 + i,
                arc_angle=48.0 + i,
                entry_angle=42.0,
                flight_time=1.4,
                peak_height=11.0
            )
            
            # Create pose sequence for this shot
            shot_poses = []
            for j in range(6):
                joints = {
                    JointType.RIGHT_SHOULDER: Joint(360, 150, 0, 0.9, JointType.RIGHT_SHOULDER, j * 0.1),
                    JointType.RIGHT_ELBOW: Joint(390, 200, 0, 0.9, JointType.RIGHT_ELBOW, j * 0.1),
                    JointType.RIGHT_WRIST: Joint(420, 250, 0, 0.9, JointType.RIGHT_WRIST, j * 0.1),
                    JointType.LEFT_HIP: Joint(300, 350, 0, 0.9, JointType.LEFT_HIP, j * 0.1),
                    JointType.RIGHT_HIP: Joint(340, 350, 0, 0.9, JointType.RIGHT_HIP, j * 0.1),
                    JointType.LEFT_ANKLE: Joint(285, 550, 0, 0.9, JointType.LEFT_ANKLE, j * 0.1),
                    JointType.RIGHT_ANKLE: Joint(355, 550, 0, 0.9, JointType.RIGHT_ANKLE, j * 0.1)
                }
                pose = ShootingPose(
                    joints=joints,
                    confidence=0.85,
                    timestamp=j * 0.1,
                    dominant_hand="right"
                )
                shot_poses.append(pose)
            
            # Analyze biomechanics
            form_score = self.biomech_analyzer.analyze_shooting_form(
                shot_poses, {"arc_angle": trajectory.arc_angle}
            )
            
            # Create shot event
            shot = ShotEvent(
                shot_successful=(i % 2 == 0),
                trajectory=trajectory,
                form_analysis=FormAnalysis(
                    overall_score=form_score.overall_score,
                    elbow_alignment=form_score.elbow_alignment_score,
                    follow_through=form_score.follow_through_score,
                    balance=form_score.balance_score,
                    arc_consistency=form_score.arc_consistency_score
                ),
                detection_confidence=0.85
            )
            shots.append(shot)
            poses.extend(shot_poses)
        
        # Analyze performance
        session_start = datetime.now() - timedelta(hours=1)
        session_end = datetime.now()
        
        performance_metrics = self.performance_analyzer.analyze_session(
            shots, session_start, session_end
        )
        
        # Verify integration results
        assert performance_metrics.total_shots == 5
        assert performance_metrics.form_score_average > 0
        assert len(performance_metrics.shot_chart_data) == 5
        assert len(self.biomech_analyzer.form_history) == 5
        
    def test_analytics_data_consistency(self):
        """Test consistency between different analytics components"""
        # Create test data with known characteristics
        points = [
            TrajectoryPoint(x=130 + j * 15, y=190 - j * 8, timestamp=j * 0.15) 
            for j in range(7)
        ]
        
        trajectory = ShotTrajectory(
            points=points,
            release_point=points[0] if points else None,
            apex_point=points[3] if len(points) > 3 else None,
            landing_point=points[-1] if points else None,
            initial_velocity=18.0,
            shot_distance=18.0,
            arc_angle=49.0,  # Optimal range
            entry_angle=43.0,
            flight_time=1.5,
            peak_height=12.0
        )
        
        # Create good form pose
        joints = {
            JointType.RIGHT_SHOULDER: Joint(360, 150, 0, 0.9, JointType.RIGHT_SHOULDER, 0.0),
            JointType.RIGHT_ELBOW: Joint(390, 200, 0, 0.9, JointType.RIGHT_ELBOW, 0.0),
            JointType.RIGHT_WRIST: Joint(420, 250, 0, 0.9, JointType.RIGHT_WRIST, 0.0),
            JointType.LEFT_HIP: Joint(300, 350, 0, 0.9, JointType.LEFT_HIP, 0.0),
            JointType.RIGHT_HIP: Joint(340, 350, 0, 0.9, JointType.RIGHT_HIP, 0.0)
        }
        pose = ShootingPose(joints=joints, timestamp=0.0, confidence=0.85, dominant_hand="right")
        
        # Analyze biomechanics
        form_score = self.biomech_analyzer.analyze_shooting_form(
            [pose], {"arc_angle": 49.0}
        )
        
        # Create shot with good characteristics
        shot = ShotEvent(
            shot_successful=True,
            trajectory=trajectory,
            form_analysis=FormAnalysis(
                overall_score=form_score.overall_score,
                elbow_alignment=form_score.elbow_alignment_score,
                balance=form_score.balance_score
            ),
            detection_confidence=0.90
        )
        
        # Analyze performance
        session_start = datetime.now() - timedelta(hours=1)
        session_end = datetime.now()
        
        metrics = self.performance_analyzer.analyze_session(
            [shot], session_start, session_end
        )
        
        # Verify consistency
        assert metrics.shooting_percentage == 100.0  # Single successful shot
        assert metrics.average_arc_angle == 49.0  # Matches trajectory
        assert metrics.form_score_average == form_score.overall_score


if __name__ == "__main__":
    pytest.main([__file__]) 