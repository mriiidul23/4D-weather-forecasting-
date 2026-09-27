# Extreme Weather Monitoring - SIH 26078

## 1. Problem
Extreme weather anomalies (like extreme rainfall, cyclones) cause catastrophic damage. Medium-range forecasts have high uncertainty in location, intensity, and timing.

## 2. Proposed Solution
An AI-driven spatio-temporal tracking system that identifies anomalies, tracks their evolution using Graph Neural Networks (GNNs), estimates ensemble uncertainty, and conditionally downscales the result to a higher regional resolution.

## 3. Architecture
1. **Preprocessing & Climatology**: Historical data defines threshold percentiles (e.g., 99th).
2. **Baseline Detection**: Scipy connected components extract anomalous weather objects.
3. **GNN Tracking**: A Lightweight PyTorch GNN associates detected blobs across timesteps, robust to merge/split operations where baseline Euclidean tracking fails.
4. **Ensemble Uncertainty**: Consolidates 10+ ensemble paths to produce probability and spatial spread.
5. **Downscaling**: Regional extraction and high-res interpolation.

## 4. Technology Stack
- **Python 3.14** (via `uv`)
- **PyTorch** (GNN Tracking)
- **SciPy / NumPy** (Thresholding, Connected Components)
- **Streamlit / Matplotlib** (Dashboard, Visualizations)

## 5. Synthetic-Data Limitation
**Important Note:** This repository currently operates in **Prototype Mode**. All weather events are generated synthetically using `SyntheticWeatherGenerator`. This allows us to validate the software architecture, tracking logic, and ensemble probability code without downloading terabytes of NCMRWF/ERA5 data on a student machine. Real data will be swapped in during the next development phase via `RealWeatherDataLoader`.

## 6. How the Model Works
- The data is transformed into standardized anomalies using our baseline climatology.
- Anomalies above the 95th percentile become binary masks.
- `BaselineDetector` extracts objects (Centroid, Area, Intensity).
- `GNNTracker` embeds these properties along with node velocity and scores bipartite affinities between time $t$ and $t+1$. The Hungarian algorithm assigns tracks.

## 7. How to Run
We use `uv` for fast dependency management.
```bash
# Run the Interactive Dashboard
uv run --with streamlit --with matplotlib --with torch --with numpy --with scipy -m streamlit run app.py

# Run the backend inference pipeline directly
uv run --with torch --with numpy --with scipy run_pipeline.py
```

## 8. Evaluation Metrics
Our `Evaluator` class measures:
- **Mean Centroid Error**: Distance between true centroid and predicted centroid.
- **IoU**: Intersection over Union of the spatial bounding boxes.
- **Track Continuity**: The percentage of the event's lifecycle successfully tracked.

## 9. Baseline Comparison
In **Scenario A** (single event), the Baseline Euclidean Tracker performs perfectly (100% Continuity).
In **Scenario B** (merging events), the Baseline Tracker drops to <40% Continuity. The GNN is designed to resolve this ambiguity through message passing and momentum embeddings.

## 10. Limitations
- GNN is currently pre-trained on a simplified synthetic spatial task.
- Downscaling is currently a Bilinear Interpolation baseline.

## 11. Future Real-Data Integration
The `SyntheticDataLoader` exposes a tensor of shape `(ensembles, time, lat, lon, features)`. We will replace this with a `RealWeatherDataLoader` that parses Xarray/NetCDF files from the NCMRWF NEPS-G dataset. The entire downstream pipeline will remain identical.
