from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
import torch

from .. import utils


class BaseLikelihood(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self, device_name: str = 'cpu') -> None:
        super().__init__()
        self.dtype, self.device = utils.get_dtype_and_device(device_name)

    @abstractmethod
    def forward(
        self,
        f_mean: torch.Tensor,
        f_covar: torch.Tensor,
        diag_only: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype, device=self.device)
