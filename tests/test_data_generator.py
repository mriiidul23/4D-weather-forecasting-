import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data.synthetic_generator import SyntheticDataLoader, SyntheticWeatherGenerator

def test_scenario_a_dimensions_and_features():
    loader = SyntheticDataLoader(scenario="A")
    low_res, high_res, gt = loader.load_data()
    
    # Tensor dimensions
    assert low_res.shape == (10, 40, 32, 32, 5), "Low-res tensor dimensions mismatch"
    assert high_res.shape == (10, 40, 64, 64, 5), "High-res tensor dimensions mismatch"
    
    # Feature dimensions
    assert low_res.shape[-1] == 5, "Feature count should be 5"
    assert high_res.shape[-1] == 5, "Feature count should be 5"
    
    # Ground truth separation
    assert len(gt) == 40, "Should have 1 ground truth per time step"

def test_nan_and_infinite_values():
    loader = SyntheticDataLoader(scenario="A")
    low_res, high_res, _ = loader.load_data()
    
    assert not np.isnan(low_res).any(), "Found NaN in low_res data"
    assert not np.isnan(high_res).any(), "Found NaN in high_res data"
    
    assert not np.isinf(low_res).any(), "Found Inf in low_res data"
    assert not np.isinf(high_res).any(), "Found Inf in high_res data"

def test_ensemble_diversity():
    loader = SyntheticDataLoader(scenario="A")
    low_res, _, _ = loader.load_data()
    
    # Ensemble members should not be perfectly identical
    diff = np.abs(low_res[0] - low_res[1]).sum()
    assert diff > 0, "Ensemble members are perfectly identical"

def test_temporal_evolution_and_movement():
    loader = SyntheticDataLoader(scenario="A")
    _, _, gt = loader.load_data()
    
    # Check that centroid moves
    t0_lat, t0_lon = gt[0].centroid_lat, gt[0].centroid_lon
    t1_lat, t1_lon = gt[-1].centroid_lat, gt[-1].centroid_lon
    
    assert t0_lat != t1_lat or t0_lon != t1_lon, "Event did not move over time"

def test_valid_bounding_boxes():
    loader = SyntheticDataLoader(scenario="A")
    _, _, gt = loader.load_data()
    
    for record in gt:
        min_lat, min_lon, max_lat, max_lon = record.bbox
        assert min_lat <= max_lat, "Invalid bounding box lat"
        assert min_lon <= max_lon, "Invalid bounding box lon"

if __name__ == "__main__":
    pytest.main([__file__])
