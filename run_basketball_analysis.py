#!/usr/bin/env python3
"""
Basketball Shot Analysis Runner
Clean interface for analyzing basketball shooting videos
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from demo import main as run_demo


def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🏀 BASKETBALL SHOT ANALYSIS - swishr AI")
    print("=" * 60)
    print("Transform any video into professional shooting analysis")
    print()


def print_analysis_summary(results_file: str):
    """Print a formatted summary of analysis results"""
    try:
        if not os.path.exists(results_file):
            print("❌ Results file not found")
            return
            
        with open(results_file, 'r') as f:
            data = json.load(f)
            
        stats = data.get('session_stats', {})
        shots = data.get('shots', [])
        
        print("\n" + "=" * 50)
        print("📊 ANALYSIS RESULTS SUMMARY")
        print("=" * 50)
        
        # Basic statistics
        total_shots = stats.get('total_shots', 0)
        shooting_pct = stats.get('shooting_percentage', 0)
        makes = stats.get('makes', 0)
        misses = stats.get('misses', 0)
        
        print(f"🎯 Total Shots Detected: {total_shots}")
        print(f"🏆 Shooting Percentage: {shooting_pct:.1f}%")
        print(f"✅ Makes: {makes}")
        print(f"❌ Misses: {misses}")
        
        # Form analysis
        if 'avg_overall_form_score' in stats:
            print(f"\n📈 SHOOTING FORM ANALYSIS")
            print(f"   Overall Form Score: {stats['avg_overall_form_score']:.1f}%")
            print(f"   Elbow Alignment: {stats['avg_elbow_alignment']:.1f}%")
            print(f"   Follow-through: {stats['avg_follow_through']:.1f}%")
            print(f"   Balance: {stats['avg_balance']:.1f}%")
            print(f"   Arc Consistency: {stats['avg_arc_consistency']:.1f}%")
            print(f"   Release Timing: {stats['avg_release_timing']:.1f}%")
            
        # Confidence metrics
        if 'avg_confidence' in stats:
            print(f"\n🔍 DETECTION CONFIDENCE")
            print(f"   Average: {stats['avg_confidence']:.1f}%")
            print(f"   Range: {stats['min_confidence']:.1f}% - {stats['max_confidence']:.1f}%")
            
        # Individual shots summary
        if shots:
            print(f"\n📋 INDIVIDUAL SHOTS")
            for i, shot in enumerate(shots[:5], 1):  # Show first 5 shots
                shot_id = shot.get('shot_id', f'shot_{i}')
                confidence = shot.get('detection_confidence', 0) * 100
                successful = shot.get('shot_successful')
                result = "✅ MAKE" if successful else "❌ MISS" if successful is False else "❓ UNKNOWN"
                print(f"   {i}. {shot_id}: {result} (Confidence: {confidence:.1f}%)")
                
            if len(shots) > 5:
                print(f"   ... and {len(shots) - 5} more shots")
                
        print("\n" + "=" * 50)
        print(f"📁 Full results saved to: {results_file}")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")


def get_recommendations(stats: dict) -> list:
    """Generate coaching recommendations based on analysis"""
    recommendations = []
    
    if 'avg_arc_consistency' in stats and stats['avg_arc_consistency'] < 50:
        recommendations.append("🎯 Focus on consistent shot arc - aim for 45-50 degree trajectory")
        
    if 'avg_elbow_alignment' in stats and stats['avg_elbow_alignment'] < 70:
        recommendations.append("💪 Work on elbow alignment - keep shooting elbow under the ball")
        
    if 'avg_follow_through' in stats and stats['avg_follow_through'] < 70:
        recommendations.append("👋 Improve follow-through - snap wrist down after release")
        
    if 'avg_balance' in stats and stats['avg_balance'] < 70:
        recommendations.append("⚖️ Focus on balance - maintain stable base throughout shot")
        
    if 'shooting_percentage' in stats and stats['shooting_percentage'] < 40:
        recommendations.append("🔄 Practice more repetitions to improve muscle memory")
        
    return recommendations


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Basketball Shot Analysis')
    
    parser.add_argument(
        'video_file',
        help='Path to basketball video file (MP4, AVI, etc.)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='data',
        help='Output directory for results (default: data)'
    )
    
    parser.add_argument(
        '--display',
        action='store_true',
        help='Show video analysis in real-time (default: no display)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.video_file):
        print(f"❌ Error: Video file not found: {args.video_file}")
        return 1
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    video_name = Path(args.video_file).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"{video_name}_analysis_{timestamp}.json")
    
    print_banner()
    print(f"🎬 Analyzing video: {args.video_file}")
    print(f"📊 Results will be saved to: {results_file}")
    print(f"🔍 Detection confidence threshold: {args.confidence}")
    print()
    
    # Prepare arguments for demo script
    demo_args = [
        '--input', args.video_file,
        '--save-data', results_file,
        '--confidence-threshold', str(args.confidence)
    ]
    
    if not args.display:
        demo_args.append('--no-display')
        
    # Run analysis
    print("🚀 Starting basketball shot analysis...")
    print("⏳ This may take a few minutes depending on video length...")
    print()
    
    # Temporarily replace sys.argv to pass arguments to demo
    original_argv = sys.argv
    sys.argv = ['demo.py'] + demo_args
    
    try:
        result = run_demo()
        
        if result == 0:
            print("\n✅ Analysis completed successfully!")
            print_analysis_summary(results_file)
            
            # Load results for recommendations
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    stats = data.get('session_stats', {})
                    
                recommendations = get_recommendations(stats)
                if recommendations:
                    print("\n💡 COACHING RECOMMENDATIONS")
                    print("-" * 30)
                    for rec in recommendations:
                        print(f"   {rec}")
                    print()
                    
            except Exception as e:
                print(f"Note: Could not generate recommendations: {e}")
                
        else:
            print("❌ Analysis failed")
            return 1
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1
        
    finally:
        sys.argv = original_argv
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 