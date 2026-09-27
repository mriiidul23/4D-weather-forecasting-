import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import uuid

@dataclass
class EventGroundTruth:
    event_id: str
    event_type: str
    timestamp: int
    centroid_lat: float
    centroid_lon: float
    intensity: float
    velocity_lat: float
    velocity_lon: float
    bbox: Tuple[float, float, float, float]  # min_lat, min_lon, max_lat, max_lon
    
class SyntheticWeatherGenerator:
    def __init__(self, 
                 grid_size: Tuple[int, int] = (32, 32), 
                 time_steps: int = 40,
                 ensemble_members: int = 10,
                 downscale_factor: int = 2):
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.ensemble_members = ensemble_members
        self.downscale_factor = downscale_factor
        self.high_res_grid = (grid_size[0] * downscale_factor, grid_size[1] * downscale_factor)
        
    def generate_background_noise(self, shape: Tuple[int, ...]) -> np.ndarray:
        # Simple uncorrelated noise for prototype. Can be improved with Perlin noise later.
        return np.random.normal(loc=0.0, scale=1.0, size=shape)

    def _create_gaussian_blob(self, grid_shape: Tuple[int, int], center: Tuple[float, float], radius: float, intensity: float) -> np.ndarray:
        x = np.arange(0, grid_shape[1])
        y = np.arange(0, grid_shape[0])
        xx, yy = np.meshgrid(x, y)
        
        # Gaussian formula
        dist_sq = (xx - center[1])**2 + (yy - center[0])**2
        blob = intensity * np.exp(-dist_sq / (2 * radius**2))
        return blob

    def generate_scenario_a(self) -> Tuple[np.ndarray, np.ndarray, List[EventGroundTruth]]:
        """
        Scenario A: One moving extreme rainfall event.
        Returns:
            low_res_data: (ensembles, time_steps, lat, lon, features)
            high_res_data: (ensembles, time_steps, lat_hr, lon_hr, features)
            ground_truth: List of EventGroundTruth
        """
        features = 5 # precip, temp, u_wind, v_wind, pressure
        low_res = np.zeros((self.ensemble_members, self.time_steps, self.grid_size[0], self.grid_size[1], features))
        high_res = np.zeros((self.ensemble_members, self.time_steps, self.high_res_grid[0], self.high_res_grid[1], features))
        
        ground_truths = []
        
        # Base event parameters
        start_pos = (5.0, 5.0)
        velocity = (0.5, 0.5) # lat, lon per timestep
        base_intensity = 10.0
        base_radius = 3.0
        
        event_id = str(uuid.uuid4())
        
        for e in range(self.ensemble_members):
            # Introduce ensemble uncertainty
            e_start_pos = (start_pos[0] + np.random.normal(0, 0.5), start_pos[1] + np.random.normal(0, 0.5))
            e_velocity = (velocity[0] + np.random.normal(0, 0.05), velocity[1] + np.random.normal(0, 0.05))
            e_intensity = base_intensity + np.random.normal(0, 1.0)
            
            for t in range(self.time_steps):
                # Calculate current position
                curr_pos = (e_start_pos[0] + t * e_velocity[0], e_start_pos[1] + t * e_velocity[1])
                
                # Add background noise
                low_res[e, t, :, :, 0] = self.generate_background_noise(self.grid_size) * 0.1
                high_res[e, t, :, :, 0] = self.generate_background_noise(self.high_res_grid) * 0.1
                
                # Generate event (Precipitation - feature 0)
                low_res_blob = self._create_gaussian_blob(self.grid_size, curr_pos, base_radius, e_intensity)
                high_res_blob = self._create_gaussian_blob(self.high_res_grid, (curr_pos[0]*self.downscale_factor, curr_pos[1]*self.downscale_factor), base_radius*self.downscale_factor, e_intensity)
                
                low_res[e, t, :, :, 0] += low_res_blob
                high_res[e, t, :, :, 0] += high_res_blob
                
                # Record ground truth (only for member 0 to represent the "true" trajectory in this synthetic setup)
                if e == 0:
                    bbox = (
                        curr_pos[0] - base_radius, curr_pos[1] - base_radius,
                        curr_pos[0] + base_radius, curr_pos[1] + base_radius
                    )
                    gt = EventGroundTruth(
                        event_id=event_id,
                        event_type="extreme_rainfall",
                        timestamp=t,
                        centroid_lat=curr_pos[0],
                        centroid_lon=curr_pos[1],
                        intensity=e_intensity,
                        velocity_lat=e_velocity[0],
                        velocity_lon=e_velocity[1],
                        bbox=bbox
                    )
                    ground_truths.append(gt)
                    
        return low_res, high_res, ground_truths

    def generate_scenario_b(self) -> Tuple[np.ndarray, np.ndarray, List[EventGroundTruth]]:
        """
        Scenario B: Two simultaneous events approaching or moving.
        """
        features = 5
        low_res = np.zeros((self.ensemble_members, self.time_steps, self.grid_size[0], self.grid_size[1], features))
        high_res = np.zeros((self.ensemble_members, self.time_steps, self.high_res_grid[0], self.high_res_grid[1], features))
        
        ground_truths = []
        
        # Event 1 parameters
        e1_start = (5.0, 5.0)
        e1_vel = (0.3, 0.3)
        e1_intensity = 10.0
        e1_id = str(uuid.uuid4())
        
        # Event 2 parameters
        e2_start = (25.0, 25.0)
        e2_vel = (-0.3, -0.3)
        e2_intensity = 12.0
        e2_id = str(uuid.uuid4())
        
        base_radius = 3.0
        
        for e in range(self.ensemble_members):
            for t in range(self.time_steps):
                p1 = (e1_start[0] + t * e1_vel[0], e1_start[1] + t * e1_vel[1])
                p2 = (e2_start[0] + t * e2_vel[0], e2_start[1] + t * e2_vel[1])
                
                low_res[e, t, :, :, 0] = self.generate_background_noise(self.grid_size) * 0.1
                high_res[e, t, :, :, 0] = self.generate_background_noise(self.high_res_grid) * 0.1
                
                # Draw events
                low_res[e, t, :, :, 0] += self._create_gaussian_blob(self.grid_size, p1, base_radius, e1_intensity)
                low_res[e, t, :, :, 0] += self._create_gaussian_blob(self.grid_size, p2, base_radius, e2_intensity)
                
                if e == 0:
                    bbox1 = (p1[0]-base_radius, p1[1]-base_radius, p1[0]+base_radius, p1[1]+base_radius)
                    bbox2 = (p2[0]-base_radius, p2[1]-base_radius, p2[0]+base_radius, p2[1]+base_radius)
                    
                    gt1 = EventGroundTruth(e1_id, "extreme_rainfall", t, p1[0], p1[1], e1_intensity, e1_vel[0], e1_vel[1], bbox1)
                    gt2 = EventGroundTruth(e2_id, "extreme_rainfall", t, p2[0], p2[1], e2_intensity, e2_vel[0], e2_vel[1], bbox2)
                    
                    ground_truths.append(gt1)
                    ground_truths.append(gt2)
                    
        return low_res, high_res, ground_truths

class SyntheticDataLoader:
    """
    Clean interface to load data as specified in Section 43 (Real-Data Swap Architecture)
    """
    def __init__(self, scenario: str = "A"):
        self.generator = SyntheticWeatherGenerator()
        self.scenario = scenario
        
    def load_data(self):
        if self.scenario == "A":
            return self.generator.generate_scenario_a()
        elif self.scenario == "B":
            return self.generator.generate_scenario_b()
        # Additional scenarios B-J to be implemented here
        else:
            raise NotImplementedError(f"Scenario {self.scenario} not yet implemented.")

if __name__ == "__main__":
    loader = SyntheticDataLoader(scenario="A")
    low_res, high_res, gt = loader.load_data()
    print(f"Low Res Shape: {low_res.shape} (Ensemble, Time, Lat, Lon, Features)")
    print(f"High Res Shape: {high_res.shape}")
    print(f"Generated {len(gt)} ground truth records for Scenario A.")
    print("Example GT at T=10:", gt[10])
