"""
数据标准化模块
"""

import numpy as np
from typing import Dict, Optional, Tuple


class Normalizer:
    def __init__(
        self,
        coord_mean: Optional[np.ndarray] = None,
        coord_std: Optional[np.ndarray] = None,
        flow_mean: Optional[np.ndarray] = None,
        flow_std: Optional[np.ndarray] = None,
    ):
        self.coord_mean = coord_mean
        self.coord_std = coord_std
        self.flow_mean = flow_mean
        self.flow_std = flow_std
    
    def normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        if self.coord_mean is None or self.coord_std is None:
            return coords
        return (coords - self.coord_mean) / (self.coord_std + 1e-8)
    
    def normalize_flows(self, flows: np.ndarray) -> np.ndarray:
        if self.flow_mean is None or self.flow_std is None:
            return flows
        return (flows - self.flow_mean) / (self.flow_std + 1e-8)
    
    def denormalize_flows(self, flows_norm: np.ndarray) -> np.ndarray:
        if self.flow_mean is None or self.flow_std is None:
            return flows_norm
        return flows_norm * self.flow_std + self.flow_mean
    
    def denormalize_coords(self, coords_norm: np.ndarray) -> np.ndarray:
        if self.coord_mean is None or self.coord_std is None:
            return coords_norm
        return coords_norm * self.coord_std + self.coord_mean
    
    def compute_from_data(
        self,
        coords: np.ndarray,
        flows: np.ndarray,
    ) -> 'Normalizer':
        self.coord_mean = coords.mean(axis=(0, 1))
        self.coord_std = coords.std(axis=(0, 1)) + 1e-8
        self.flow_mean = flows.mean(axis=(0, 1))
        self.flow_std = flows.std(axis=(0, 1)) + 1e-8
        return self
    
    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            'coord_mean': self.coord_mean,
            'coord_std': self.coord_std,
            'flow_mean': self.flow_mean,
            'flow_std': self.flow_std,
        }
    
    @classmethod
    def from_params(cls, params: Dict[str, np.ndarray]) -> 'Normalizer':
        return cls(
            coord_mean=params.get('coord_mean'),
            coord_std=params.get('coord_std'),
            flow_mean=params.get('flow_mean'),
            flow_std=params.get('flow_std'),
        )
