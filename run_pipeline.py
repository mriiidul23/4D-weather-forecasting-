import numpy as np
import sys
import os

from src.data.synthetic_generator import SyntheticDataLoader
from src.data.climatology import SyntheticClimatology
from src.models.baseline_detector import BaselineDetector
from src.models.tracker import SimpleTracker
from src.models.gnn_tracker import GNNTracker
from src.models.ensemble_uncertainty import EnsembleUncertaintyEstimator
from src.evaluation.metrics import Evaluator

def run_pipeline(scenario: str = "B"):
    print(f"--- Starting Pipeline for Scenario {scenario} ---")
    
    # 1. INPUT
    print("1. Loading Synthetic Data...")
    loader = SyntheticDataLoader(scenario=scenario)
    low_res, high_res, gt = loader.load_data()
    print(f"   Input Tensor Shape: {low_res.shape}")
    
    # 2. PREPROCESSING / CLIMATOLOGY
    print("2. Generating Climatological Baseline...")
    climatology = SyntheticClimatology()
    climatology.generate_historical_baseline()
    
    # 3. DETECTION (Baseline)
    print("3. Running Anomaly Detection (Baseline)...")
    detector = BaselineDetector(climatology, feature_idx=0, percentile=95)
    
    # 4 & 5. TRACKING
    print("4 & 5. Tracking Objects over Time...")
    
    # 6. ENSEMBLE UNCERTAINTY
    print("6. Calculating Ensemble Uncertainty...")
    all_ensemble_tracks = {}
    num_ensembles = low_res.shape[0]
    
    # Determine which tracker to use
    model_path = "gnn_tracker.pth"
    use_gnn = False # Set to True when GNN pretraining handles motion
    if use_gnn and os.path.exists(model_path):
        print("   Using GNN Tracker")
    else:
        print("   Using Simple Tracker")
        use_gnn = False
        
    for e in range(num_ensembles):
        if use_gnn:
            tracker = GNNTracker(model_path=model_path)
        else:
            tracker = SimpleTracker(max_distance=5.0)
            
        for t in range(low_res.shape[1]):
            objects = detector.detect(low_res[e, t], time_step=t)
            tracker.update(objects, time_step=t)
        all_ensemble_tracks[e] = tracker.get_all_tracks()
        
    valid_tracks = [tr for tr in all_ensemble_tracks[0] if tr.duration >= 5]
    print(f"   Found {len(valid_tracks)} significant tracks in Control Member (duration >= 5).")
    
    uncertainty_estimator = EnsembleUncertaintyEstimator()
    uncertainty_metrics = uncertainty_estimator.estimate_uncertainty(all_ensemble_tracks, target_event_idx=0)
    for k, v in uncertainty_metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")

    # 7. REGIONAL EXTRACTION & DOWNSCALING (Placeholder)
    print("7. Regional Downscaling (Placeholder)...")
    
    # 8. RESULTS & EVALUATION
    print("8. Evaluating against Ground Truth (Control Member)...")
    evaluator = Evaluator(gt, valid_tracks)
    metrics = evaluator.evaluate_localization()
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")

if __name__ == "__main__":
    # Test Scenario A first to ensure GNN works on basic tasks
    run_pipeline(scenario="A")
    print("\n" + "="*50 + "\n")
    # Then test Scenario B
    run_pipeline(scenario="B")
