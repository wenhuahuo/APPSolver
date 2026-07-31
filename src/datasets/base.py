"""
数据集基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import torch


class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def get_normalization_params(self) -> Dict[str, Any]:
        pass
