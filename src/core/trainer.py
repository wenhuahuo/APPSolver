"""
训练器核心模块
"""

import os
import time
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# from ..models.base import ModelRegistry, count_parameters


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'manual_seed'):
        torch.backends.mps.manual_seed(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: AdamW,
    epochs: int,
    warmup_epochs: int,
    steps_per_epoch: int
) -> LambdaLR:
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_metrics(
    gt_field: np.ndarray,
    pred_field: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple:
    gt_field = gt_field[valid_mask]
    pred_field = pred_field[valid_mask]

    mse = np.mean((gt_field - pred_field) ** 2)
    mae = np.mean(np.abs(gt_field - pred_field))
    rmse = np.sqrt(mse)

    return mse, mae, rmse


def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    best_loss: float,
    args,
    norm_params: Dict,
    is_best: bool = False,
    filepath: str = 'checkpoint.pt',
):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'args': vars(args) if hasattr(args, '__dict__') else {},
        'norm_params': norm_params,
    }
    
    torch.save(checkpoint, filepath)
    print(f"  Checkpoint saved: {filepath}")
    
    if is_best:
        best_path = os.path.join(os.path.dirname(filepath) or '.', 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"  Best model saved: {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[AdamW] = None,
    scheduler: Optional[LambdaLR] = None,
):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('best_loss', float('inf'))
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"  Resumed from epoch {checkpoint['epoch']}, best_loss={best_loss:.6f}")
    return start_epoch, best_loss


class IrregularTrainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: AdamW,
        scheduler: LambdaLR,
        device: torch.device,
        args,
        norm_params: Optional[Dict] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args
        self.norm_params = norm_params
        self.scaler = GradScaler() if args.use_amp and device.type == 'cuda' else None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for step, (pos, fx, y) in enumerate(self.train_loader):
            pos = pos.to(self.device)
            fx = fx.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            if self.args.use_amp and self.scaler is not None:
                with autocast():
                    pred = self.model(pos, fx)
                    loss = nn.MSELoss()(pred, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(pos, fx)
                loss = nn.MSELoss()(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if step % self.args.log_interval == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"  Epoch [{epoch}] Step [{step}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.6f}  LR: {current_lr:.2e}")

        avg_loss = total_loss / max(num_batches, 1)
        return {'train_loss': avg_loss}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        all_mse = []
        all_mae = []
        all_rmse = []

        for pos, fx, y in self.val_loader:
            pos = pos.to(self.device)
            fx = fx.to(self.device)
            y = y.to(self.device)

            pred = self.model(pos, fx)
            loss = nn.MSELoss()(pred, y)

            total_loss += loss.item()
            num_batches += 1

            if self.norm_params is not None:
                flow_std = torch.tensor(self.norm_params['flow_std'], device=self.device, dtype=pred.dtype)
                flow_mean = torch.tensor(self.norm_params['flow_mean'], device=self.device, dtype=pred.dtype)
                pred_denorm = pred * flow_std + flow_mean
                target_denorm = y * flow_std + flow_mean
            else:
                pred_denorm = pred
                target_denorm = y

            pred_np = pred_denorm.cpu().numpy()
            target_np = target_denorm.cpu().numpy()
            batch_size = pred_np.shape[0]

            for b in range(batch_size):
                mse, mae, rmse = compute_metrics(
                    target_np[b], pred_np[b], np.ones(pred_np.shape[1], dtype=bool)
                )
                all_mse.append(mse)
                all_mae.append(mae)
                all_rmse.append(rmse)

        avg_loss = total_loss / max(num_batches, 1)

        result = {
            'val_loss': avg_loss,
            'val_mse': float(np.mean(all_mse)) if all_mse else 0.0,
            'val_mae': float(np.mean(all_mae)) if all_mae else 0.0,
            'val_rmse': float(np.mean(all_rmse)) if all_rmse else 0.0,
        }

        return result


class PatchTrainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: AdamW,
        scheduler: LambdaLR,
        device: torch.device,
        args,
        norm_params: Optional[Dict] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args
        self.norm_params = norm_params
        self.scaler = GradScaler() if args.use_amp and device.type == 'cuda' else None
        self.coord_dim = 2
    
    def compute_mse_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            pred = pred * mask.unsqueeze(1).float()
            target = target * mask.unsqueeze(1).float()
            loss = ((pred - target) ** 2).sum() / (mask.sum() + 1e-8)
        else:
            loss = nn.functional.mse_loss(pred, target)
        return loss
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(self.train_loader):
            input_patches = batch['input'].to(self.device)
            target_patches = batch['output'].to(self.device)
            mask = batch['mask'].to(self.device)

            target_flow = target_patches[:, self.coord_dim:self.coord_dim + self.args.output_dim, :, :]

            self.optimizer.zero_grad()

            if self.args.use_amp and self.scaler is not None:
                with autocast():
                    pred = self.model(input_patches, mask)
                    loss = self.compute_mse_loss(pred, target_flow, mask)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(input_patches, mask)
                loss = self.compute_mse_loss(pred, target_flow, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if step % self.args.log_interval == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"  Epoch [{epoch}] Step [{step}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.6f}  LR: {current_lr:.2e}")

        avg_loss = total_loss / max(num_batches, 1)
        return {'train_loss': avg_loss}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        all_mse = []
        all_mae = []
        all_rmse = []

        for batch in self.val_loader:
            input_patches = batch['input'].to(self.device)
            target_patches = batch['output'].to(self.device)
            mask = batch['mask'].to(self.device)

            target_flow = target_patches[:, self.coord_dim:self.coord_dim + self.args.output_dim, :, :]

            pred = self.model(input_patches, mask)
            loss = self.compute_mse_loss(pred, target_flow, mask)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        result = {
            'val_loss': avg_loss,
        }

        return result
