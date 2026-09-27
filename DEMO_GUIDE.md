# SIH 26078: 5-Minute Demonstration Guide

*This script is designed for the student team to read/adapt during the Smart India Hackathon evaluation.*

---

## 0:00 - 1:00: The Problem & The UI
**"Welcome to our prototype for AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies."**
*Action: Have the Streamlit app (`app.py`) open on the main monitor.*

"Medium-range ensemble forecasts are massive and difficult to parse manually. Our system automatically ingests 10+ ensemble forecasts, identifies extreme anomalies like rainfall or cyclones, and tracks their exact path. 
For this demonstration, to ensure it runs smoothly on this local machine, we are using a **Synthetic Data Prototype**. It simulates the exact tensor structure of NCMRWF data."

## 1:00 - 2:00: Baseline vs GNN
*Action: Select 'Scenario B - Multiple Events' from the left sidebar.*

"Here on the main map, you can see two simulated extreme rainfall events. 
When we use a standard thresholding baseline, the tracker gets confused when these two storms get close and merge. If you look at the **Baseline vs GNN Comparison** table on the right, you'll see the Baseline's Track Continuity drops.
However, our **PyTorch GNN** embeds the storms' velocity, area, and intensity into nodes, allowing it to correctly maintain track identity even during complex merges."

## 2:00 - 3:00: Ensemble Uncertainty
*Action: Point to the 'Ensemble Uncertainty' panel.*

"We don't just rely on the deterministic forecast. The model runs this tracking algorithm across all ensemble members. 
As you can see here, it aggregates the tracks to give us an **Event Probability** and explicitly calculates the **Spatial Spread** in degrees. This tells disaster management exactly how confident we are in the storm's trajectory."

## 3:00 - 4:00: Downscaling (Phase 9)
*Action: Scroll down to the Downscaling Prototype section.*

"Global forecasts often run at 12km resolution. We built a regional extraction pipeline that crops the bounding box of the anomaly and passes it to a downscaling module. Right now, this is showing our Bilinear Upscaling baseline versus the synthetic high-resolution target, but the architecture allows us to slot in a learned diffusion/super-resolution model seamlessly."

## 4:00 - 5:00: Architecture & Real Data Plan
*Action: Drag the Timeline Slider to show the events moving.*

"The entire pipeline from Anomaly Detection -> GNN Tracking -> Uncertainty Estimation runs in under a minute. 
Because we isolated the Data Loader behind a strict interface, replacing this synthetic data with real NetCDF files from ERA5 or NCMRWF will require zero changes to our Tracking or GNN models. 
Thank you."
