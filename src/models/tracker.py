import numpy as np
from typing import List, Dict, Optional
import math
import uuid
from dataclasses import dataclass

try:
    from .baseline_detector import DetectedObject
except ImportError:
    from baseline_detector import DetectedObject

@dataclass
class TrackedEvent:
    event_id: str
    start_time: int
    end_time: int
    history: List[DetectedObject]
    
    @property
    def current_lat(self) -> float:
        return self.history[-1].centroid_lat
        
    @property
    def current_lon(self) -> float:
        return self.history[-1].centroid_lon
        
    @property
    def duration(self) -> int:
        return self.end_time - self.start_time + 1
        
    @property
    def speed_and_direction(self) -> tuple[float, float]:
        """Returns (speed, direction_in_degrees) based on the last 2 steps if available."""
        if len(self.history) < 2:
            return 0.0, 0.0
        
        p1 = self.history[-2]
        p2 = self.history[-1]
        
        dlat = p2.centroid_lat - p1.centroid_lat
        dlon = p2.centroid_lon - p1.centroid_lon
        
        speed = math.sqrt(dlat**2 + dlon**2)
        direction = math.degrees(math.atan2(dlat, dlon))
        return speed, direction

class SimpleTracker:
    def __init__(self, max_distance: float = 5.0):
        """
        max_distance: The maximum Euclidean distance a centroid can move in one time step
        to be considered the same event.
        """
        self.max_distance = max_distance
        self.active_tracks: List[TrackedEvent] = []
        self.finished_tracks: List[TrackedEvent] = []

    def update(self, detected_objects: List[DetectedObject], time_step: int):
        if not self.active_tracks:
            # All objects start new tracks
            for obj in detected_objects:
                new_track = TrackedEvent(
                    event_id=str(uuid.uuid4()),
                    start_time=time_step,
                    end_time=time_step,
                    history=[obj]
                )
                self.active_tracks.append(new_track)
            return

        unassigned_objects = list(detected_objects)
        next_active_tracks = []

        # Simple greedy assignment
        for track in self.active_tracks:
            best_obj = None
            best_dist = float('inf')
            
            for obj in unassigned_objects:
                dist = math.sqrt((track.current_lat - obj.centroid_lat)**2 + 
                                 (track.current_lon - obj.centroid_lon)**2)
                if dist < best_dist and dist <= self.max_distance:
                    best_dist = dist
                    best_obj = obj
            
            if best_obj:
                # Assign object to this track
                track.history.append(best_obj)
                track.end_time = time_step
                next_active_tracks.append(track)
                unassigned_objects.remove(best_obj)
            else:
                # Track dies
                self.finished_tracks.append(track)

        # Unassigned objects become new tracks
        for obj in unassigned_objects:
            new_track = TrackedEvent(
                event_id=str(uuid.uuid4()),
                start_time=time_step,
                end_time=time_step,
                history=[obj]
            )
            next_active_tracks.append(new_track)
            
        self.active_tracks = next_active_tracks
        
    def get_all_tracks(self) -> List[TrackedEvent]:
        return self.finished_tracks + self.active_tracks

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
    from data.synthetic_generator import SyntheticDataLoader
    from data.climatology import SyntheticClimatology
    from models.baseline_detector import BaselineDetector
    
    loader = SyntheticDataLoader(scenario="A")
    low_res, _, gt = loader.load_data()
    
    climatology = SyntheticClimatology()
    climatology.generate_historical_baseline()
    
    detector = BaselineDetector(climatology, feature_idx=0, percentile=95) # 95th for better detection of the edge
    tracker = SimpleTracker(max_distance=3.0)
    
    # Process the first ensemble member over time
    for t in range(loader.generator.time_steps):
        forecast_frame = low_res[0, t]
        objects = detector.detect(forecast_frame, time_step=t)
        tracker.update(objects, time_step=t)
        
    all_tracks = tracker.get_all_tracks()
    print(f"Total unique tracks identified: {len(all_tracks)}")
    
    # Find the longest track
    if all_tracks:
        longest = max(all_tracks, key=lambda x: x.duration)
        print(f"Longest track duration: {longest.duration} time steps.")
        print(f"Last known position: ({longest.current_lat:.2f}, {longest.current_lon:.2f})")
        speed, direction = longest.speed_and_direction
        print(f"Final speed: {speed:.2f}, direction: {direction:.2f} degrees")
        print(f"Ground Truth Final Position: ({gt[-1].centroid_lat:.2f}, {gt[-1].centroid_lon:.2f})")
