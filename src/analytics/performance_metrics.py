"""
Performance Metrics and Analytics Module

Implements comprehensive shooting performance analysis including:
- Shooting percentage calculations
- Form consistency scoring
- Distance-based performance analysis
- Shot chart generation and analysis
- Trend analysis and improvement tracking
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, Counter
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from computer_vision.shot_detection import ShotEvent
from computer_vision.trajectory_analysis import ShotTrajectory


@dataclass
class ShootingMetrics:
    """Comprehensive shooting performance metrics"""
    
    # Basic shooting statistics
    total_shots: int = 0
    made_shots: int = 0
    shooting_percentage: float = 0.0
    
    # Distance-based metrics
    shots_by_distance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    average_shot_distance: float = 0.0
    
    # Arc and trajectory metrics
    average_arc_angle: float = 0.0
    arc_consistency: float = 0.0  # Standard deviation of arc angles
    average_entry_angle: float = 0.0
    entry_angle_consistency: float = 0.0
    
    # Form consistency metrics
    release_point_consistency: float = 0.0
    form_score_average: float = 0.0
    form_score_consistency: float = 0.0
    
    # Time-based metrics
    session_date: datetime = field(default_factory=datetime.now)
    session_duration: float = 0.0  # minutes
    shots_per_minute: float = 0.0
    
    # Shot chart data
    shot_chart_data: List[Dict[str, Any]] = field(default_factory=list)
    
    # Trend data
    improvement_rate: float = 0.0  # percentage improvement over time
    consistency_trend: float = 0.0  # positive = improving consistency


@dataclass 
class ShotChartPoint:
    """Individual shot for shot chart visualization"""
    x: float  # Court position x
    y: float  # Court position y
    distance: float  # Distance from basket
    angle: float  # Angle relative to basket
    made: bool  # Shot outcome
    confidence: float  # Detection confidence
    form_score: float  # Overall form score
    timestamp: datetime
    

class PerformanceAnalyzer:
    """
    Advanced performance metrics calculation and analysis
    
    Calculates comprehensive shooting statistics, analyzes trends,
    generates shot charts, and provides detailed performance insights.
    """
    
    def __init__(self, 
                 distance_zones: Optional[Dict[str, Tuple[float, float]]] = None,
                 form_weight_factors: Optional[Dict[str, float]] = None):
        """
        Initialize performance analyzer
        
        Args:
            distance_zones: Custom distance zones for analysis
            form_weight_factors: Weights for different form components
        """
        # Default distance zones (in feet from basket)
        self.distance_zones = distance_zones or {
            "close": (0, 8),      # Under the basket to close range
            "mid_range": (8, 16), # Mid-range shots
            "three_point": (16, 26), # Three-point range
            "long_range": (26, 40)   # Deep three-pointers
        }
        
        # Form component weights for overall scoring
        self.form_weights = form_weight_factors or {
            "elbow_alignment": 0.25,
            "arc_angle": 0.20,
            "follow_through": 0.25,
            "balance": 0.30
        }
        
        # Historical data storage
        self.shot_history: List[ShotEvent] = []
        self.session_history: List[ShootingMetrics] = []
        
    def analyze_session(self, 
                       shots: List[ShotEvent],
                       session_start: datetime,
                       session_end: datetime) -> ShootingMetrics:
        """
        Analyze a complete shooting session
        
        Args:
            shots: List of shot events from the session
            session_start: Session start time
            session_end: Session end time
            
        Returns:
            Comprehensive shooting metrics for the session
        """
        if not shots:
            return ShootingMetrics(session_date=session_start)
            
        metrics = ShootingMetrics(session_date=session_start)
        
        # Basic shooting statistics
        metrics.total_shots = len(shots)
        metrics.made_shots = sum(1 for shot in shots if shot.shot_successful)
        metrics.shooting_percentage = (metrics.made_shots / metrics.total_shots * 100) if metrics.total_shots > 0 else 0.0
        
        # Session timing
        session_duration = (session_end - session_start).total_seconds() / 60  # minutes
        metrics.session_duration = session_duration
        metrics.shots_per_minute = metrics.total_shots / session_duration if session_duration > 0 else 0.0
        
        # Distance-based analysis
        metrics.shots_by_distance = self._analyze_distance_performance(shots)
        distances = [shot.trajectory.shot_distance for shot in shots if shot.trajectory and shot.trajectory.shot_distance]
        metrics.average_shot_distance = statistics.mean(distances) if distances else 0.0
        
        # Arc and trajectory analysis
        arc_angles = [shot.trajectory.arc_angle for shot in shots if shot.trajectory and shot.trajectory.arc_angle]
        entry_angles = [shot.trajectory.entry_angle for shot in shots if shot.trajectory and shot.trajectory.entry_angle]
        
        if arc_angles:
            metrics.average_arc_angle = statistics.mean(arc_angles)
            metrics.arc_consistency = statistics.stdev(arc_angles) if len(arc_angles) > 1 else 0.0
            
        if entry_angles:
            metrics.average_entry_angle = statistics.mean(entry_angles)
            metrics.entry_angle_consistency = statistics.stdev(entry_angles) if len(entry_angles) > 1 else 0.0
        
        # Form consistency analysis
        form_scores = [shot.form_analysis.overall_score for shot in shots if shot.form_analysis]
        release_points = [(shot.release_point.x, shot.release_point.y) for shot in shots if shot.release_point]
        
        if form_scores:
            metrics.form_score_average = statistics.mean(form_scores)
            metrics.form_score_consistency = statistics.stdev(form_scores) if len(form_scores) > 1 else 0.0
            
        if release_points:
            metrics.release_point_consistency = self._calculate_release_point_consistency(release_points)
        
        # Generate shot chart data
        metrics.shot_chart_data = self._generate_shot_chart_data(shots)
        
        # Calculate trends if we have historical data
        if self.session_history:
            metrics.improvement_rate = self._calculate_improvement_rate(metrics)
            metrics.consistency_trend = self._calculate_consistency_trend(metrics)
        
        # Store session for future trend analysis
        self.session_history.append(metrics)
        self.shot_history.extend(shots)
        
        return metrics
    
    def _analyze_distance_performance(self, shots: List[ShotEvent]) -> Dict[str, Dict[str, Any]]:
        """Analyze shooting performance by distance zones"""
        zone_stats = {}
        
        for zone_name, (min_dist, max_dist) in self.distance_zones.items():
            zone_shots = [
                shot for shot in shots 
                if shot.trajectory and shot.trajectory.shot_distance 
                and min_dist <= shot.trajectory.shot_distance <= max_dist
            ]
            
            if zone_shots:
                made = sum(1 for shot in zone_shots if shot.shot_successful)
                total = len(zone_shots)
                percentage = (made / total * 100) if total > 0 else 0.0
                
                # Calculate average form score for this zone
                form_scores = [shot.form_analysis.overall_score for shot in zone_shots if shot.form_analysis]
                avg_form_score = statistics.mean(form_scores) if form_scores else 0.0
                
                # Calculate average arc angle for this zone
                arc_angles = [shot.trajectory.arc_angle for shot in zone_shots if shot.trajectory.arc_angle]
                avg_arc_angle = statistics.mean(arc_angles) if arc_angles else 0.0
                
                zone_stats[zone_name] = {
                    "total_shots": total,
                    "made_shots": made,
                    "shooting_percentage": percentage,
                    "average_form_score": avg_form_score,
                    "average_arc_angle": avg_arc_angle,
                    "distance_range": (min_dist, max_dist)
                }
            else:
                zone_stats[zone_name] = {
                    "total_shots": 0,
                    "made_shots": 0,
                    "shooting_percentage": 0.0,
                    "average_form_score": 0.0,
                    "average_arc_angle": 0.0,
                    "distance_range": (min_dist, max_dist)
                }
        
        return zone_stats
    
    def _calculate_release_point_consistency(self, release_points: List[Tuple[float, float]]) -> float:
        """Calculate consistency of release point positions"""
        if len(release_points) < 2:
            return 100.0  # Perfect consistency with single point
            
        # Calculate centroid
        centroid_x = statistics.mean([point[0] for point in release_points])
        centroid_y = statistics.mean([point[1] for point in release_points])
        
        # Calculate average distance from centroid
        distances = [
            np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2) 
            for x, y in release_points
        ]
        
        avg_distance = statistics.mean(distances)
        
        # Convert to consistency score (lower distance = higher consistency)
        # Normalize to 0-100 scale where 100 is perfect consistency
        max_reasonable_distance = 50  # pixels
        consistency = max(0, (max_reasonable_distance - avg_distance) / max_reasonable_distance * 100)
        
        return consistency
    
    def _generate_shot_chart_data(self, shots: List[ShotEvent]) -> List[Dict[str, Any]]:
        """Generate shot chart visualization data"""
        shot_chart = []
        
        for shot in shots:
            if not shot.trajectory or not shot.trajectory.shot_distance:
                continue
                
            # Calculate position relative to basket (simplified court coordinates)
            # In a real implementation, this would use court detection data
            distance = shot.trajectory.shot_distance
            angle = getattr(shot.trajectory, 'angle_from_basket', 0)  # Would need to add this
            
            # Convert polar coordinates to cartesian for visualization
            x_pos = distance * np.cos(np.radians(angle))
            y_pos = distance * np.sin(np.radians(angle))
            
            shot_point = {
                "x": x_pos,
                "y": y_pos,
                "distance": distance,
                "angle": angle,
                "made": shot.shot_successful,
                "confidence": shot.detection_confidence,
                "form_score": shot.form_analysis.overall_score if shot.form_analysis else 0.0,
                "arc_angle": shot.trajectory.arc_angle if shot.trajectory.arc_angle else 0.0,
                "timestamp": shot.timestamp.isoformat() if shot.timestamp else None
            }
            
            shot_chart.append(shot_point)
        
        return shot_chart
    
    def _calculate_improvement_rate(self, current_metrics: ShootingMetrics) -> float:
        """Calculate improvement rate compared to historical performance"""
        if len(self.session_history) < 2:
            return 0.0
            
        # Compare with sessions from the last 7 days for trend
        recent_cutoff = current_metrics.session_date - timedelta(days=7)
        recent_sessions = [
            session for session in self.session_history[-10:]  # Last 10 sessions max
            if session.session_date >= recent_cutoff
        ]
        
        if len(recent_sessions) < 2:
            return 0.0
            
        # Calculate average improvement across multiple metrics
        old_avg_percentage = statistics.mean([s.shooting_percentage for s in recent_sessions[:-1]])
        old_avg_form = statistics.mean([s.form_score_average for s in recent_sessions[:-1]])
        
        current_percentage = current_metrics.shooting_percentage
        current_form = current_metrics.form_score_average
        
        # Calculate percentage improvement
        percentage_improvement = ((current_percentage - old_avg_percentage) / old_avg_percentage * 100) if old_avg_percentage > 0 else 0
        form_improvement = ((current_form - old_avg_form) / old_avg_form * 100) if old_avg_form > 0 else 0
        
        # Weighted average of improvements
        overall_improvement = (percentage_improvement * 0.6 + form_improvement * 0.4)
        
        return overall_improvement
    
    def _calculate_consistency_trend(self, current_metrics: ShootingMetrics) -> float:
        """Calculate trend in shooting consistency over time"""
        if len(self.session_history) < 3:
            return 0.0
            
        # Get consistency metrics from recent sessions
        recent_sessions = self.session_history[-5:]  # Last 5 sessions
        
        # Calculate consistency scores (lower variance = higher consistency)
        consistency_scores = []
        for session in recent_sessions:
            # Combine multiple consistency metrics
            form_consistency = 100 - session.form_score_consistency  # Invert so higher is better
            arc_consistency = 100 - session.arc_consistency
            release_consistency = session.release_point_consistency
            
            overall_consistency = (form_consistency + arc_consistency + release_consistency) / 3
            consistency_scores.append(overall_consistency)
        
        # Calculate trend using linear regression slope
        if len(consistency_scores) >= 2:
            x = list(range(len(consistency_scores)))
            slope = np.polyfit(x, consistency_scores, 1)[0]
            return slope  # Positive = improving consistency
        
        return 0.0
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive performance summary over specified period
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Performance summary with trends and insights
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_sessions = [
            session for session in self.session_history 
            if session.session_date >= cutoff_date
        ]
        
        if not recent_sessions:
            return {"error": "No sessions found in specified period"}
        
        # Aggregate metrics
        total_shots = sum(session.total_shots for session in recent_sessions)
        total_made = sum(session.made_shots for session in recent_sessions)
        overall_percentage = (total_made / total_shots * 100) if total_shots > 0 else 0.0
        
        # Average metrics
        avg_form_score = statistics.mean([s.form_score_average for s in recent_sessions if s.form_score_average > 0])
        avg_arc_angle = statistics.mean([s.average_arc_angle for s in recent_sessions if s.average_arc_angle > 0])
        
        # Identify strengths and weaknesses by distance
        distance_performance = defaultdict(list)
        for session in recent_sessions:
            for zone, stats in session.shots_by_distance.items():
                if stats['total_shots'] > 0:
                    distance_performance[zone].append(stats['shooting_percentage'])
        
        zone_averages = {
            zone: statistics.mean(percentages) 
            for zone, percentages in distance_performance.items()
        }
        
        best_zone = max(zone_averages.items(), key=lambda x: x[1]) if zone_averages else ("none", 0)
        worst_zone = min(zone_averages.items(), key=lambda x: x[1]) if zone_averages else ("none", 0)
        
        return {
            "period_days": days,
            "total_sessions": len(recent_sessions),
            "total_shots": total_shots,
            "overall_shooting_percentage": overall_percentage,
            "average_form_score": avg_form_score,
            "average_arc_angle": avg_arc_angle,
            "best_distance_zone": {"zone": best_zone[0], "percentage": best_zone[1]},
            "worst_distance_zone": {"zone": worst_zone[0], "percentage": worst_zone[1]},
            "distance_zone_performance": dict(zone_averages),
            "improvement_trend": statistics.mean([s.improvement_rate for s in recent_sessions[-5:] if s.improvement_rate]) if any(s.improvement_rate for s in recent_sessions[-5:]) else 0.0,
            "consistency_trend": statistics.mean([s.consistency_trend for s in recent_sessions[-5:] if s.consistency_trend]) if any(s.consistency_trend for s in recent_sessions[-5:]) else 0.0
        }
    
    def export_metrics(self, filename: str, format: str = "json") -> bool:
        """
        Export performance metrics to file
        
        Args:
            filename: Output filename
            format: Export format ("json", "csv")
            
        Returns:
            Success status
        """
        try:
            if format.lower() == "json":
                data = {
                    "session_history": [
                        {
                            "session_date": session.session_date.isoformat(),
                            "total_shots": session.total_shots,
                            "shooting_percentage": session.shooting_percentage,
                            "form_score_average": session.form_score_average,
                            "average_arc_angle": session.average_arc_angle,
                            "shots_by_distance": session.shots_by_distance,
                            "shot_chart_data": session.shot_chart_data
                        }
                        for session in self.session_history
                    ]
                }
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            elif format.lower() == "csv":
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Date", "Total Shots", "Made Shots", "Shooting %", 
                        "Form Score", "Arc Angle", "Session Duration"
                    ])
                    
                    for session in self.session_history:
                        writer.writerow([
                            session.session_date.strftime("%Y-%m-%d %H:%M"),
                            session.total_shots,
                            session.made_shots,
                            f"{session.shooting_percentage:.1f}%",
                            f"{session.form_score_average:.1f}",
                            f"{session.average_arc_angle:.1f}°",
                            f"{session.session_duration:.1f} min"
                        ])
            
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False 