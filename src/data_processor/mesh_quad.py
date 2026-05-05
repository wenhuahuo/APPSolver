"""
Quadtree Mesh 划分模块

将二维网格点递归划分为四叉树结构，每个叶子节点包含约 patch_size 个点。
支持距离场自适应细化和下采样。
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class QuadNode:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    points: np.ndarray
    level: int
    original_points: Optional[np.ndarray] = None
    children: Optional[List['QuadNode']] = None
    is_leaf: bool = True
    patch_id: int = -1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max)
    
    @property
    def point_count(self) -> int:
        return len(self.points)


class QuadTreeMesh:
    def __init__(
        self,
        grid_coords: np.ndarray,
        patch_size: int = 64,
        ship_length: float = 7.0,
        ref_point: Tuple[float, float] = (3.0, 0.0),
        distance_threshold_1: float = 1.0,
        distance_threshold_2: float = 1.5,
        enable_distance_refine: bool = False
    ):
        if grid_coords.shape[1] != 2:
            raise ValueError(f"grid_coords必须是[N, 2]形状，当前为{grid_coords.shape}")
        
        self.grid_coords = grid_coords
        self.patch_size = patch_size
        self.ship_length = ship_length
        self.ref_point = ref_point
        self.distance_threshold_1 = distance_threshold_1 * ship_length
        self.distance_threshold_2 = distance_threshold_2 * ship_length
        self.enable_distance_refine = enable_distance_refine
        self.distance_field: Optional[np.ndarray] = None
        
        self.root: Optional[QuadNode] = None
        self.patches: List[QuadNode] = []
        self._patch_counter = 0
        self.max_depth = 64
        self.min_cell_size = 1e-10
        
        if enable_distance_refine:
            self._compute_distance_field()
        
        self._build_tree()
    
    def _compute_distance_field(self) -> np.ndarray:
        ref_x, ref_y = self.ref_point
        distances = np.sqrt(
            (self.grid_coords[:, 0] - ref_x) ** 2 +
            (self.grid_coords[:, 1] - ref_y) ** 2
        )
        self.distance_field = distances
        return distances
    
    def _get_target_patch_size_for_node(self, node: QuadNode) -> int:
        if not self.enable_distance_refine:
            return self.patch_size
        
        center_x, center_y = node.center
        ref_x, ref_y = self.ref_point
        distance = np.sqrt((center_x - ref_x) ** 2 + (center_y - ref_y) ** 2)
        
        if distance < self.distance_threshold_1:
            return max(4, self.patch_size // 4)
        elif distance < self.distance_threshold_2:
            return max(4, self.patch_size // 2)
        else:
            return self.patch_size
    
    def _build_tree(self) -> None:
        x_min, x_max = self.grid_coords[:, 0].min(), self.grid_coords[:, 0].max()
        y_min, y_max = self.grid_coords[:, 1].min(), self.grid_coords[:, 1].max()
        
        padding = 1e-6
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        
        all_indices = np.arange(len(self.grid_coords))
        self.root = QuadNode(
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            points=all_indices,
            level=0
        )
        
        self._split_node(self.root)
    
    def _split_node(self, node: QuadNode) -> None:
        target_patch_size = self._get_target_patch_size_for_node(node)

        width = node.x_max - node.x_min
        height = node.y_max - node.y_min

        if node.level >= self.max_depth or width <= self.min_cell_size or height <= self.min_cell_size:
            node.is_leaf = True
            node.patch_id = self._patch_counter
            self._patch_counter += 1
            self.patches.append(node)
            return
        
        if len(node.points) <= target_patch_size:
            node.is_leaf = True
            node.patch_id = self._patch_counter
            self._patch_counter += 1
            self.patches.append(node)
            return
        
        node.is_leaf = False
        x_center, y_center = node.center
        
        quadrants = [[], [], [], []]
        
        for idx in node.points:
            x, y = self.grid_coords[idx]
            if x <= x_center and y <= y_center:
                quadrants[0].append(idx)
            elif x > x_center and y <= y_center:
                quadrants[1].append(idx)
            elif x <= x_center and y > y_center:
                quadrants[2].append(idx)
            else:
                quadrants[3].append(idx)
        
        node.children = []

        non_empty_quadrants = [q for q in quadrants if q]
        if len(non_empty_quadrants) <= 1:
            node.is_leaf = True
            node.patch_id = self._patch_counter
            self._patch_counter += 1
            self.patches.append(node)
            node.children = None
            return
        
        if quadrants[0]:
            child = QuadNode(
                x_min=node.x_min, x_max=x_center,
                y_min=node.y_min, y_max=y_center,
                points=np.array(quadrants[0]),
                level=node.level + 1
            )
            node.children.append(child)
            self._split_node(child)
        
        if quadrants[1]:
            child = QuadNode(
                x_min=x_center, x_max=node.x_max,
                y_min=node.y_min, y_max=y_center,
                points=np.array(quadrants[1]),
                level=node.level + 1
            )
            node.children.append(child)
            self._split_node(child)
        
        if quadrants[2]:
            child = QuadNode(
                x_min=node.x_min, x_max=x_center,
                y_min=y_center, y_max=node.y_max,
                points=np.array(quadrants[2]),
                level=node.level + 1
            )
            node.children.append(child)
            self._split_node(child)
        
        if quadrants[3]:
            child = QuadNode(
                x_min=x_center, x_max=node.x_max,
                y_min=y_center, y_max=node.y_max,
                points=np.array(quadrants[3]),
                level=node.level + 1
            )
            node.children.append(child)
            self._split_node(child)
    
    def downsample_patches_by_distance(
        self,
        method: str = 'uniform',
        target_points: Optional[int] = None,
        min_points: int = 16
    ) -> None:
        if self.distance_field is None:
            self._compute_distance_field()
        
        if not self.patches:
            print("警告：没有可用的patches")
            return
        
        if target_points is None:
            detected_min = min(len(patch.points) for patch in self.patches)
            target_points = max(detected_min, min_points)
        
        if method == 'uniform':
            self._downsample_uniform(target_points, min_points)
        elif method == 'distance':
            self._downsample_by_distance(target_points, min_points)
        else:
            raise ValueError(f"未知的下采样方法: {method}")
    
    def _downsample_uniform(self, target_points: int, min_points: int = 16) -> None:
        if target_points < min_points:
            target_points = min_points
        
        max_points = max(len(patch.points) for patch in self.patches)
        if target_points >= max_points:
            return
        
        for patch in self.patches:
            current_points = len(patch.points)
            if current_points > target_points:
                patch.original_points = patch.points.copy()
                indices = patch.points
                step = max(1, current_points // target_points)
                sampled_indices = indices[::step][:target_points]
                patch.points = sampled_indices
    
    def _downsample_by_distance(self, target_points: int, min_points: int = 16) -> None:
        if target_points < min_points:
            target_points = min_points
        
        ref_x, ref_y = self.ref_point
        
        for patch in self.patches:
            current_points = len(patch.points)
            if current_points <= target_points:
                continue
            
            patch_center_x, patch_center_y = patch.center
            distance_to_ship = np.sqrt(
                (patch_center_x - ref_x) ** 2 + 
                (patch_center_y - ref_y) ** 2
            )
            
            if distance_to_ship < self.distance_threshold_1:
                continue
            
            patch.original_points = patch.points.copy()
            indices = patch.points
            step = max(1, current_points // target_points)
            sampled_indices = indices[::step][:target_points]
            patch.points = sampled_indices
        
        final_min = min(len(patch.points) for patch in self.patches)
        if final_min != target_points:
            self._downsample_uniform(target_points, min_points)
