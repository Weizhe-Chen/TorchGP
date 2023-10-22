from typing import Tuple

import torch
from torch.nn import Parameter

from .base_likelihood import BaseLikelihood
from .. import utils


class GaussianLikelihood(BaseLikelihood):
    def __init__(self, noise_variance: float, device_name: str = "cpu") -> None:
        super().__init__(device_name)
        self.noise_variance = noise_variance

    def forward(
        self,
        f_mean: torch.Tensor,
        f_covar: torch.Tensor,
        diag_only: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if diag_only:
            f_covar += self.noise_variance
        else:
            f_covar.diagonal().add_(self.noise_variance)
        return f_mean, f_covar

    @property
    def noise_variance(self) -> torch.Tensor:
        return utils.free_to_constrained(self.free_noise_variance)

    @noise_variance.setter
    def noise_variance(self, value):
        self.free_noise_variance = Parameter(
            utils.constrained_to_free(self._to_tensor(value))
        )
