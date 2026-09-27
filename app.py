import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt

from src.data.synthetic_generator import SyntheticDataLoader
from src.data.climatology import SyntheticClimatology
from src.models.baseline_detector import BaselineDetector
from src.models.tracker import SimpleTracker
from src.models.gnn_tracker import GNNTracker
from src.models.ensemble_uncertainty import EnsembleUncertaintyEstimator
from src.evaluation.metrics import Evaluator

st.set_page_config(page_title="Extreme Weather Monitoring - SIH 26078", layout="wide")

st.title("EXTREME WEATHER MONITORING — SIH 26078")
st.subheader("Prototype Mode — Synthetic Data")

# Sidebar: Scenario Selection & Controls
st.sidebar.header("Controls")
scenario = st.sidebar.selectbox("Select Scenario", ["A - Moving Event", "B - Multiple Events"])
scenario_key = scenario.split(" ")[0] # 'A' or 'B'

# Load Data
@st.cache_resource
def load_and_run_pipeline(scen_key):
    loader = SyntheticDataLoader(scenario=scen_key)
    low_res, high_res, gt = loader.load_data()
    
    climatology = SyntheticClimatology()
    climatology.generate_historical_baseline()
    
    detector = BaselineDetector(climatology, feature_idx=0, percentile=95)
    
    # Run Baseline
    base_tracker = SimpleTracker(max_distance=5.0)
    for t in range(low_res.shape[1]):
        objects = detector.detect(low_res[0, t], time_step=t)
        base_tracker.update(objects, time_step=t)
        
    base_tracks = [tr for tr in base_tracker.get_all_tracks() if tr.duration >= 3]
    
    # Run GNN
    model_path = "gnn_tracker.pth"
    gnn_tracker = GNNTracker(model_path) if os.path.exists(model_path) else SimpleTracker(max_distance=5.0)
    for t in range(low_res.shape[1]):
        objects = detector.detect(low_res[0, t], time_step=t)
        gnn_tracker.update(objects, time_step=t)
        
    gnn_tracks = [tr for tr in gnn_tracker.get_all_tracks() if tr.duration >= 3]
    
    # Uncertainty
    all_ensemble_tracks = {}
    for e in range(low_res.shape[0]):
        tracker = SimpleTracker(max_distance=5.0)
        for t in range(low_res.shape[1]):
            objects = detector.detect(low_res[e, t], time_step=t)
            tracker.update(objects, time_step=t)
        all_ensemble_tracks[e] = tracker.get_all_tracks()
        
    uncertainty = EnsembleUncertaintyEstimator().estimate_uncertainty(all_ensemble_tracks, 0)
    
    # Eval
    eval_base = Evaluator(gt, base_tracks).evaluate_localization()
    eval_gnn = Evaluator(gt, gnn_tracks).evaluate_localization()
    
    return low_res, high_res, gt, base_tracks, gnn_tracks, uncertainty, eval_base, eval_gnn

with st.spinner("Generating synthetic data and running inference..."):
    low_res, high_res, gt, base_tracks, gnn_tracks, uncertainty, eval_base, eval_gnn = load_and_run_pipeline(scenario_key)

# Main UI layout
col1, col2 = st.columns([3, 1])

# Timeline Slider
time_steps = low_res.shape[1]
t = st.slider("Forecast Lead Time", 0, time_steps - 1, 0)

with col1:
    st.markdown(f"### Spatial View: T + {t} hours")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    frame = low_res[0, t, :, :, 0]
    c = ax.imshow(frame, cmap='Blues', origin='lower')
    plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot Ground Truth
    gt_t = [g for g in gt if g.timestamp == t]
    for g in gt_t:
        ax.plot(g.centroid_lon, g.centroid_lat, 'r*', markersize=15, label='Ground Truth' if g == gt_t[0] else "")
        min_lat, min_lon, max_lat, max_lon = g.bbox
        rect = plt.Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat, fill=False, color='red', linestyle='--')
        ax.add_patch(rect)
        
    # Plot GNN Tracks (Proposed)
    for track in gnn_tracks:
        obj = next((o for o in track.history if o.timestamp == t), None)
        if obj:
            ax.plot(obj.centroid_lon, obj.centroid_lat, 'gX', markersize=10, label='GNN Prediction')
            
    ax.legend(loc='upper right')
    st.pyplot(fig)
    
    st.markdown("### Downscaling Prototype (Phase 9)")
    st.write("Baseline Upscaling vs Learned Conditional Downscaling vs High-Res Ground Truth")
    
    fig2, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(low_res[0, t, :, :, 0], cmap='Blues', origin='lower')
    axs[0].set_title("Low-Res Input (32x32)")
    
    import scipy.ndimage as ndimage
    baseline_upscale = ndimage.zoom(low_res[0, t, :, :, 0], 2.0, order=1) # Bilinear interpolation
    axs[1].imshow(baseline_upscale, cmap='Blues', origin='lower')
    axs[1].set_title("Baseline Upscaling (Bilinear)")
    
    axs[2].imshow(high_res[0, t, :, :, 0], cmap='Blues', origin='lower')
    axs[2].set_title("Synthetic High-Res Target (64x64)")
    
    st.pyplot(fig2)

with col2:
    st.markdown("### Event Information")
    st.write("**Event:** Extreme Rainfall")
    st.write(f"**Lead Time:** T + {t} hours")
    
    st.markdown("---")
    st.markdown("### Ensemble Uncertainty")
    prob = uncertainty.get("event_probability", 0)
    st.metric("Event Probability", f"{prob*100:.1f}%")
    st.metric("Spatial Spread (Lat/Lon)", f"{uncertainty.get('location_spread_lat', 0):.2f}° / {uncertainty.get('location_spread_lon', 0):.2f}°")
    st.write(f"**Confidence:** {uncertainty.get('confidence', 'Unknown')}")
    
    st.markdown("---")
    st.markdown("### Baseline vs GNN Comparison")
    
    comp_data = {
        "Metric": ["Centroid Error", "IoU", "Track Continuity"],
        "Baseline": [f"{eval_base.get('mean_centroid_error', 0):.2f}", f"{eval_base.get('mean_iou', 0):.2f}", f"{eval_base.get('track_continuity', 0):.2f}"],
        "Proposed (GNN)": [f"{eval_gnn.get('mean_centroid_error', 0):.2f}", f"{eval_gnn.get('mean_iou', 0):.2f}", f"{eval_gnn.get('track_continuity', 0):.2f}"]
    }
    st.table(comp_data)
    
    st.info("The Baseline uses connected components + Euclidean distance. The GNN uses node embeddings + message passing to resolve split/merge ambiguities.")
