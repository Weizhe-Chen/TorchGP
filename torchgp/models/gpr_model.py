from typing import Tuple

import numpy as np
import torch

from .. import utils
from ..kernels import BaseKernel
from ..likelihoods import GaussianLikelihood
from .base_model import BaseModel


class GPRModel(BaseModel):
    def __init__(
        self,
        kernel: BaseKernel,
        noise_variance: float,
        device_name: str = "cpu",
    ) -> None:
        likelihood = GaussianLikelihood(noise_variance)
        super().__init__(kernel, likelihood, device_name)

    def common_terms(self):
        Kyy = self.kernel(self.train_x)  # O(N^2)
        Kyy.diagonal().add_(self.likelihood.noise_variance)  # O(N)
        TrilKyy = utils.tril(Kyy)  # O(N^3)  -> bottleneck
        InvKyy_y = utils.tril_solve(TrilKyy, self.train_y)  # O(N^2)
        return TrilKyy, InvKyy_y

    def evidence(self) -> torch.Tensor:
        TrilKyy, InvKyy_y = self.common_terms()  # O(N^3)
        quadratic_term = (self.train_y * InvKyy_y).sum()  # O(N)
        log_term = utils.tril_logdet(TrilKyy)  #  O(N)
        constant_term = len(self.train_y) * np.log(2 * np.pi)  # O(1)
        return -0.5 * (quadratic_term + log_term + constant_term)

    def forward(
        self,
        test_x: torch.Tensor,
        diag_only: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        TrilKyy, InvKyy_y = self.common_terms()  # O(N^3)
        Ksf = self.kernel(test_x, self.train_x)  # O(S N)
        InvTrilKyy_Kfs = utils.tril_inv_matmul(TrilKyy, Ksf.T)  # O(N^2 S)
        mean = Ksf @ InvKyy_y  # O(S N)
        if diag_only:
            kss = self.kernel.diag(test_x)  # O(S)
            var = kss - utils.tril_diag_quadratic(InvTrilKyy_Kfs)  # O(S N)
            return mean, var
        else:
            Kss = self.kernel(test_x)
            cov = Kss - InvTrilKyy_Kfs.T @ InvTrilKyy_Kfs  # O(S^2 N)
            return mean, cov
