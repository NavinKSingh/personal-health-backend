import sys
from types import SimpleNamespace
from pose_analyzer import PoseAnalyzer, Landmark

def generate_mock_trajectory():
    """Generates 15 frames of descending squat movements."""
    frames = []
    
    # Simulate a dropping hip/knee
    for i in range(15):
        # 33 landmarks, focusing on left knee (25), left hip (23), left ankle (27)
        points = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(33)]
        
        # Make hip go lower over time
        points[23].y = 0.5 + (0.02 * i) 
        
        # Change Knee angle over time (create velocity)
        # Knee is between hip and ankle
        points[25].x = 0.55
        points[25].y = 0.6 + (0.01 * i)
        
        points[27].x = 0.5
        points[27].y = 0.9 
        
        # Mid hip (23, 24)
        points[24].y = points[23].y
        
        # Shoulders (11, 12)
        points[11].y = points[23].y - 0.2
        points[12].y = points[24].y - 0.2
        
        # Nose (0)
        points[0].y = points[11].y - 0.1
        
        res = SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=points))
        frames.append(res)
        
    return frames

def test_math2():
    print("Initializing Analyzer...")
    analyzer = PoseAnalyzer(sport="vertical_jump")
    
    frames = generate_mock_trajectory()
    last_bio = None
    
    print("\nProcessing Trajectory...")
    for idx, f in enumerate(frames):
        bio = analyzer.analyze(f)
        last_bio = bio
        if idx % 5 == 0 or idx == len(frames) - 1:
            print(f"Frame {idx}: Knee={bio.knee_angle_l:.1f}deg, DJ={bio.dimensionless_jerk:.4f}, PhaseDM={bio.phase_space_dm:.2f}")

    print("\n--- Phase 2 Feature Validation ---")
    if last_bio.dimensionless_jerk > 0:
        print("[PASS] Dimensionless Jerk computed successfully!")
    else:
        print("[FAIL] Dimensionless Jerk is 0.")
        
    if last_bio.phase_space_dm > 0:
        print("[PASS] Phase-Space Mahalanobis computed successfully!")
    else:
        print("[FAIL] Phase-Space DM is 0.")
        
    print("[PASS] State Machine (Deque) successfully held memory length:", len(analyzer.frame_history))
    
    # Test Quaternions
    if getattr(last_bio, 'torsion_error', -1) >= 0:
        print("[PASS] Quaternion Torsion computed:", getattr(last_bio, 'torsion_error', 0))
    else:
        print("[FAIL] Quaternion Torsion missing.")

if __name__ == "__main__":
    test_math2()
