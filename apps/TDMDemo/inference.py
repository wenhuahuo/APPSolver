"""Real APP-Transformer single-step and autoregressive ShipBench inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.core.metrics import patches_to_points, recover_points_knn
from src.datasets.shipBench import PatchFlowFieldDataset
from src.models.patch.transformer import Transformer

from .backend import CHANNELS, checkpoint_dir, condition_path


@dataclass
class PredictionResult:
    coords: np.ndarray
    current: np.ndarray
    prediction: np.ndarray
    target: np.ndarray
    metrics: dict[str, float | list[float]]
    latency_seconds: float
    source_step: int
    target_step: int


class APPPredictor:
    """Load one canonical per-hull APP model and run it on a chosen condition."""

    def __init__(self, condition: str, device_name: str = "auto") -> None:
        self.condition = condition
        self.hull = condition.split("/", maxsplit=1)[0]
        self.run_dir = checkpoint_dir(self.hull, "app_transformer")
        checkpoint = self.run_dir / "model_best_mae.pth"
        if not checkpoint.is_file():
            checkpoint = self.run_dir / "model_final.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"APP checkpoint not found: {self.run_dir}")

        stats_path = self.run_dir / "normalization_stats.npz"
        if not stats_path.is_file():
            raise FileNotFoundError(f"Normalization stats not found: {stats_path}")
        with np.load(stats_path) as stats:
            self.normalization = {
                key: stats[key].astype(np.float32)
                for key in ("coord_mean", "coord_std", "flow_mean", "flow_std")
            }

        self.dataset = PatchFlowFieldDataset(
            data_dir=str(condition_path(condition)),
            patch_size=256,
            enable_downsample=True,
            downsample_method="distance",
            downsample_ratio=0.6,
            normalize=True,
            split="all",
            rollout_holdout_steps=0,
            normalization_params=self.normalization,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        in_dim = int(state["input_proj.weight"].shape[1])
        out_dim = int(state["output_proj.weight"].shape[0])
        d_model = int(state["input_proj.weight"].shape[0])
        max_patches = int(state["pos_embed"].shape[1])
        num_layers = len(
            {
                key.split(".")[2]
                for key in state
                if key.startswith("transformer.layers.")
            }
        )
        self.global_points = in_dim // 6
        if out_dim != self.global_points * len(CHANNELS):
            raise ValueError("Checkpoint dimensions do not match four-channel ShipBench data")
        if self.dataset.max_points > self.global_points:
            raise ValueError("Selected condition has more patch points than the checkpoint")

        if device_name == "auto":
            if torch.cuda.is_available():
                device_name = "cuda"
            elif torch.backends.mps.is_available():
                device_name = "mps"
            else:
                device_name = "cpu"
        self.device = torch.device(device_name)
        self.model = Transformer(
            in_flattened_dim=in_dim,
            out_flattened_dim=out_dim,
            d_model=d_model,
            nhead=4,
            num_layers=num_layers,
            max_patches=max_patches,
            params_dim=None,
        )
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @property
    def frame_count(self) -> int:
        return int(self.dataset.num_timesteps)

    def _normalize_coords(self, values: np.ndarray) -> np.ndarray:
        return (values - self.normalization["coord_mean"]) / self.normalization["coord_std"]

    def _normalize_flow(self, values: np.ndarray) -> np.ndarray:
        return (values - self.normalization["flow_mean"]) / self.normalization["flow_std"]

    def _denormalize_flow(self, values: np.ndarray) -> np.ndarray:
        return values * self.normalization["flow_std"] + self.normalization["flow_mean"]

    def _make_input(self, coords: np.ndarray, flow: np.ndarray) -> torch.Tensor:
        values = np.concatenate(
            [self._normalize_coords(coords), self._normalize_flow(flow)], axis=1
        ).astype(np.float32, copy=False)
        patches = np.zeros(
            (self.dataset.num_patches, self.global_points, 6), dtype=np.float32
        )
        for patch_index, patch in enumerate(self.dataset.quadtree.patches):
            point_ids = patch.points
            patches[patch_index, : len(point_ids)] = values[point_ids]
        return torch.from_numpy(patches.reshape(self.dataset.num_patches, -1))

    def _recover(self, prediction: torch.Tensor) -> np.ndarray:
        local_width = self.dataset.max_points * len(CHANNELS)
        sampled, _ = patches_to_points(
            prediction[:, :, :local_width],
            self.dataset.quadtree,
            batch_size=1,
            n_points=self.dataset.num_points,
            n_channels=len(CHANNELS),
            input_dim=len(CHANNELS),
            max_points=self.dataset.max_points,
        )
        recovered = recover_points_knn(
            sampled,
            self.dataset.recovery_indices,
            self.dataset.recovery_weights,
            self.dataset.sampled_indices,
        )
        return recovered[0].numpy()

    @staticmethod
    def _metrics(prediction_norm: np.ndarray, target_norm: np.ndarray) -> dict:
        error = prediction_norm - target_norm
        mse_channels = np.mean(error**2, axis=0)
        mae_channels = np.mean(np.abs(error), axis=0)
        relative_l2 = float(
            np.linalg.norm(error) / max(np.linalg.norm(target_norm), 1e-12)
        )
        return {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "relative_l2": relative_l2,
            "mae_per_channel": mae_channels.tolist(),
            "rmse_per_channel": np.sqrt(mse_channels).tolist(),
        }

    @torch.inference_mode()
    def predict_flow(self, coords: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, float]:
        model_input = self._make_input(coords, current).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        predicted_patches = self.model(model_input, params_embed=None).cpu()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()
        prediction_norm = self._recover(predicted_patches)
        elapsed = time.perf_counter() - started
        return self._denormalize_flow(prediction_norm), elapsed

    def predict_step(self, source_step: int) -> PredictionResult:
        if not 0 <= source_step < self.frame_count - 1:
            raise IndexError("Source timestep must have a following target frame")
        coords = self.dataset._all_coords[source_step]
        current = self.dataset._all_flows[source_step]
        target = self.dataset._all_flows[source_step + 1]
        prediction, latency = self.predict_flow(coords, current)
        metrics = self._metrics(
            self._normalize_flow(prediction), self._normalize_flow(target)
        )
        return PredictionResult(
            coords=coords,
            current=current,
            prediction=prediction,
            target=target,
            metrics=metrics,
            latency_seconds=latency,
            source_step=source_step,
            target_step=source_step + 1,
        )

    def rollout(self, source_step: int, horizon: int) -> tuple[list[PredictionResult], float]:
        if horizon < 1 or source_step + horizon >= self.frame_count:
            raise ValueError("Rollout horizon exceeds the available ShipBench sequence")
        coords = self.dataset._all_coords[source_step]
        current = self.dataset._all_flows[source_step]
        results: list[PredictionResult] = []
        total_latency = 0.0
        for offset in range(1, horizon + 1):
            target = self.dataset._all_flows[source_step + offset]
            prediction, latency = self.predict_flow(coords, current)
            total_latency += latency
            results.append(
                PredictionResult(
                    coords=coords,
                    current=current,
                    prediction=prediction,
                    target=target,
                    metrics=self._metrics(
                        self._normalize_flow(prediction), self._normalize_flow(target)
                    ),
                    latency_seconds=latency,
                    source_step=source_step + offset - 1,
                    target_step=source_step + offset,
                )
            )
            current = prediction
        return results, total_latency
