from typing import Tuple

import numpy as np
import torch
from torch.nn import Parameter
from tqdm import tqdm

from .. import utils
from ..kernels import BaseKernel
from .gpr_model import GPRModel


class SGPRModel(GPRModel):
    def __init__(
        self,
        x_pseudo: np.ndarray,
        kernel: BaseKernel,
        noise_variance: float,
        device_name: str = "cpu",
        jitter: float = 1e-6,
    ) -> None:
        super().__init__(kernel, noise_variance, device_name)
        self.pseudo_x = Parameter(self._to_tensor(x_pseudo))
        self.jitter = jitter

    @property
    def x_pseudo(self) -> np.ndarray:
        return utils.to_array(self.pseudo_x)

    @property
    @torch.inference_mode()
    def f_pseudo(self) -> np.ndarray:
        return utils.to_array(self.forward(self.pseudo_x)[0])

    @property
    def num_pseudo(self) -> int:
        return self.pseudo_x.shape[0]

    def common_terms(self):
        noise_std = self.likelihood.noise_variance.sqrt()
        Kuu = self.kernel(self.pseudo_x)
        Kuf = self.kernel(self.pseudo_x, self.train_x)  # O(M N)
        TrilKuu = utils.tril(Kuu, self.jitter)  # O(M^3)
        # A = sigma^{-1} TrilKuu^{-1} Kuf -> shape (M, N)
        A = utils.tril_inv_matmul(TrilKuu, Kuf).div(noise_std)  # O(M^2 N)
        # B = I + A A.T -> shape (M, M)
        B = torch.eye(self.num_pseudo).to(A) + A @ A.T  # O(M^2 N)
        TrilB = utils.tril(B, base_jitter=0.0)  # O(M^3)
        # c = sigma^{-1} TrilB^{-1} A y -> shape (M, 1)
        Ay = A @ self.train_y  # O(M N) -> shape (M, 1)
        c = utils.tril_inv_matmul(TrilB, Ay).div(noise_std)  # O(M^2)
        return A, TrilB, c, TrilKuu

    def evidence(self) -> torch.Tensor:
        A, TrilB, c, _ = self.common_terms()
        sigma2 = self.likelihood.noise_variance
        return 0.5 * (
            # -N log(2pi)
            -self.num_train * np.log(2 * np.pi)
            # -log|B|
            - utils.tril_logdet(TrilB)
            # -N log(sigma^{2})
            - self.num_train * torch.log(sigma2)
            # -sigma^{-2} y.T y
            - self.train_y.square().sum().div(sigma2)
            # +sigma^{-2} y.T A.T B^{-1} A y = sigma^{-2} c.T c
            + c.square().sum()
            # -sigma^{-2} trace(Kff)
            - self.kernel.diag(self.train_x).sum().div(sigma2)
            # + trace(A A.T)
            + utils.trace_quadratic(A)
        )

    def forward(
        self,
        test_x: torch.Tensor,
        diag_only: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, TrilB, c, TrilKuu = self.common_terms()
        Ksu = self.kernel(test_x, self.pseudo_x)  # O(S M)
        # Lk^{-1} Ku* -> shape (M, S)
        InvTrilKuu_Kus = utils.tril_inv_matmul(TrilKuu, Ksu.t())  # O(M^2 S)
        # Lb^{-1} Lk^{-1} Ku* -> shape (M, S)  O(M^2 S)
        InvTrilB_InvTrilKuu_Kus = utils.tril_inv_matmul(TrilB, InvTrilKuu_Kus)
        # K*u InvTrilKuu.T InvTrilB.T c -> shape (S, 1)
        mean = InvTrilB_InvTrilKuu_Kus.t() @ c  # O(S M)
        if diag_only:
            kss = self.kernel.diag(test_x)
            var = (
                kss
                - utils.tril_diag_quadratic(InvTrilKuu_Kus)
                + utils.tril_diag_quadratic(InvTrilB_InvTrilKuu_Kus)
            )
            return mean, var
        else:
            Kss = self.kernel(test_x)
            cov = (
                Kss
                - InvTrilKuu_Kus.t() @ InvTrilKuu_Kus
                + InvTrilB_InvTrilKuu_Kus.t() @ InvTrilB_InvTrilKuu_Kus
            )
            return mean, cov
