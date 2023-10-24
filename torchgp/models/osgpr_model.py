from typing import Tuple

import numpy as np
import torch
from torch.nn import Parameter
from tqdm import tqdm

from .. import utils
from ..kernels import BaseKernel
from .sgpr_model import SGPRModel


class OSGPRModel(SGPRModel):
    def __init__(
        self,
        x_pseudo: np.ndarray,
        kernel: BaseKernel,
        noise_variance: float,
        device_name: str = "cpu",
        jitter: float = 1e-6,
        keep_rate: float = 0.7,
    ) -> None:
        super().__init__(
            x_pseudo=x_pseudo,
            kernel=kernel,
            noise_variance=noise_variance,
            device_name=device_name,
            jitter=jitter,
        )
        self.keep_rate = keep_rate

    @property
    def has_memory(self) -> bool:
        return self.zu is not None

    def add_data(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        self.validate_data(x_new, y_new)
        self.memorize()
        self.train_x = self._to_tensor(x_new)
        self.train_y = self._to_tensor(y_new)
        if self.has_memory:
            self.update_pseudo_x()

    def memorize(self) -> None:
        if not hasattr(self, "train_x"):
            self.zu = None  # old pseudo inputs
            self.mu = None  # old variational mean
            self.Suu = None  # old variational covariance
            self.Kuu = None  # old kernel matrix
        else:
            zu = self.pseudo_x.data.clone()
            mu, Suu = self.forward(zu, diag_only=False)
            Kuu = self.kernel(zu)
            Kuu.diagonal().add_(self.jitter)
            self.zu = zu
            self.mu = mu.detach()
            self.Suu = Suu.detach()
            self.Kuu = Kuu.detach()

    def update_pseudo_x(self) -> None:
        num_kept = int(self.keep_rate * self.num_pseudo)
        num_added = self.num_pseudo - num_kept
        kept_idx = np.random.permutation(self.num_pseudo)[:num_kept]
        added_idx = np.random.permutation(self.num_train)[:num_added]
        self.pseudo_x = Parameter(
            torch.cat((self.zu[kept_idx], self.train_x[added_idx]))
        )

    def common_terms(self):
        if not self.has_memory:
            return super().common_terms()

        sigma2 = self.likelihood.noise_variance
        Kvv = self.kernel(self.pseudo_x)  # O(M^2)
        Kvv.diagonal().add_(self.jitter)
        Kvf = self.kernel(self.pseudo_x, self.train_x)  # O(MN)
        Kvu = self.kernel(self.pseudo_x, self.zu)  # O(MM)

        TrilKvv = utils.tril(Kvv, self.jitter)
        TrilSuu = utils.tril(self.Suu, self.jitter)
        TrilOldKuu = utils.tril(self.Kuu, self.jitter)

        InvTrilKvv_Kvf = utils.tril_inv_matmul(TrilKvv, Kvf)
        D1 = (InvTrilKvv_Kvf @ InvTrilKvv_Kvf.T).div(sigma2)
        InvTrilKvv_Kvu = utils.tril_inv_matmul(TrilKvv, Kvu)
        Kuv_InvTrilKvvT = InvTrilKvv_Kvu.T
        InvTrilSuu_Kuv_InvTrilKvvT = utils.tril_inv_matmul(TrilSuu, Kuv_InvTrilKvvT)
        D2 = InvTrilSuu_Kuv_InvTrilKvvT.T @ InvTrilSuu_Kuv_InvTrilKvvT
        InvTrilOldKuu_Kuv_InvTrilKvvT = utils.tril_inv_matmul(TrilOldKuu, Kuv_InvTrilKvvT)
        D3 = InvTrilOldKuu_Kuv_InvTrilKvvT.T @ InvTrilOldKuu_Kuv_InvTrilKvvT
        D = torch.eye(D1.shape[0]).to(D1) + D1 + D2 - D3
        D.diagonal().add_(self.jitter)
        TrilD = utils.tril(D, self.jitter)

        InvTrilSuu_Kuv = utils.tril_inv_matmul(TrilSuu, Kvu.T)
        InvTrilSuu_mu = utils.tril_inv_matmul(TrilSuu, self.mu)
        c1 = (Kvf @ self.train_y).div(sigma2)
        c2 = InvTrilSuu_Kuv.T @ InvTrilSuu_mu
        c = c1 + c2
        return c, TrilKvv, TrilD, TrilSuu, TrilOldKuu, Kvf, Kvu

    def evidence(self) -> torch.Tensor:
        if not self.has_memory:
            return super().evidence()

        c, TrilKvv, TrilD, TrilSuu, TrilOldKuu, Kvf, Kvu = self.common_terms()
        sigma2 = self.likelihood.noise_variance

        Kuu = self.kernel(self.zu)  # O(M^2)
        Kuu.diagonal().add_(self.jitter)
        kff = self.kernel.diag(self.train_x)  # O(N)

        InvTrilKvv_c = utils.tril_inv_matmul(TrilKvv, c)
        InvTrilD_InvTrilKvv_c = utils.tril_inv_matmul(TrilD, InvTrilKvv_c)
        InvTrilSuu_mu = utils.tril_inv_matmul(TrilSuu, self.mu)

        InvTrilKvv_Kvu = utils.tril_inv_matmul(TrilKvv, Kvu)
        Quu = InvTrilKvv_Kvu.T @ InvTrilKvv_Kvu
        Euu = Kuu - Quu
        InvSuu_Euu = utils.tril_solve(TrilSuu, Euu)
        InvOldKuu_Euu = utils.tril_solve(TrilOldKuu, Euu)
        InvTrilKvv_Kvf = utils.tril_inv_matmul(TrilKvv, Kvf)

        constant_term = -self.num_train * np.log(2 * np.pi)
        quadratic_terms = (
            -(self.train_y.square().sum().div(sigma2))
            + InvTrilD_InvTrilKvv_c.square().sum()
            - InvTrilSuu_mu.square().sum()
        )
        log_terms = (
            -utils.tril_logdet(TrilD)
            - utils.tril_logdet(TrilSuu)
            + utils.tril_logdet(TrilOldKuu)
            - self.num_train * torch.log(sigma2)
        )
        trace_terms = (
            -InvSuu_Euu.trace()
            + InvOldKuu_Euu.trace()
            - kff.sum().div(sigma2)
            + utils.trace_quadratic(InvTrilKvv_Kvf).div(sigma2)
        )
        return 0.5 * (constant_term + quadratic_terms + log_terms + trace_terms)

    def forward(
        self,
        test_x: torch.Tensor,
        diag_only: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.has_memory:
            return super().forward(test_x, diag_only)

        c, TrilKvv, TrilD, _, _, _, _ = self.common_terms()
        Kvs = self.kernel(self.pseudo_x, test_x)  # O(MS)

        InvTrilKvv_Kvs = utils.tril_inv_matmul(TrilKvv, Kvs)
        InvTrilD_InvTrilKvv_Kvs = utils.tril_inv_matmul(TrilD, InvTrilKvv_Kvs)
        InvTrilKvv_c = utils.tril_inv_matmul(TrilKvv, c)
        InvTrilD_InvTrilKvv_c = utils.tril_inv_matmul(TrilD, InvTrilKvv_c)

        mean = InvTrilD_InvTrilKvv_Kvs.T @ InvTrilD_InvTrilKvv_c

        if diag_only:
            kss = self.kernel.diag(test_x)
            var = (
                kss
                - utils.tril_diag_quadratic(InvTrilKvv_Kvs)
                + utils.tril_diag_quadratic(InvTrilD_InvTrilKvv_Kvs)
            )
            return mean, var
        else:
            Kss = self.kernel(test_x)
            cov = (
                Kss
                - InvTrilKvv_Kvs.T @ InvTrilKvv_Kvs
                + InvTrilD_InvTrilKvv_Kvs.T @ InvTrilD_InvTrilKvv_Kvs
            )
            return mean, cov
