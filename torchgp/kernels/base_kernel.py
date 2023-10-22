from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch.nn.parameter import Parameter

from .. import utils


class BaseKernel(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, output_scale: float, device_name: str = "cpu") -> None:
        super().__init__()
        self.dtype, self.device = utils.get_dtype_and_device(device_name)
        self.output_scale = output_scale

    @property
    def output_scale(self) -> torch.Tensor:
        return utils.free_to_constrained(self.free_output_scale)

    @output_scale.setter
    def output_scale(self, value):
        self.free_output_scale = Parameter(
            utils.constrained_to_free(self._to_tensor(value))
        )

    def diag(self, x):
        return self.output_scale * torch.ones(x.shape[0], 1).to(x)

    @abstractmethod
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute the covariance matrix between two sets of inputs.

        Parameters:
            x1 (torch.Tensor): First input tensor of shape
                (num_inputs_1, dim_inputs_1).
            x2 (torch.Tensor): Second input tensor of shape
                (num_inputs_2, dim_inputs_2).

        Returns:
            cov_mat (torch.Tensor): Full covariance matrix of shape
                (num_inputs_1, num_inputs_2).

        Raises:
            NotImplementedError: This is an abstract method and must be
                implemented by a subclass.

        """
        raise NotImplementedError

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype, device=self.device)
