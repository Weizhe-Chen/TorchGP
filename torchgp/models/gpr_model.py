from typing import Tuple

import numpy as np
import torch
from torch.nn import Parameter
from tqdm import tqdm

from .. import utils
from ..kernels import BaseKernel
from ..likelihoods import GaussianLikelihood
from .base_model import BaseModel


class GPRModel(BaseModel):

    def __init__(
        self,
        kernel: BaseKernel,
        noise_variance: float,
        device_name: str,
    ) -> None:
        likelihood = GaussianLikelihood(noise_variance)
        super().__init__(kernel, likelihood, device_name)

    def learn(
        self,
        x_new: np.ndarray,
        y_new: np.ndarray,
        optimizer: str = "l-bfgs-b",
        num_steps: int = 100,
        verbose: bool = False,
    ) -> None:
        self.add_data(x_new, y_new)

        self.train()
        if optimizer == "l-bfgs-b":
            self._learn_with_lbfgsb(num_steps, verbose)
        elif optimizer == "adam":
            self._learn_with_adam(num_steps, verbose)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        self.eval()

    def _learn_with_lbfgsb(self, num_steps: int, verbose: bool = False):
        from pytorch_minimize.optim import MinimizeWrapper
        optimizer = MinimizeWrapper(
            self.parameters(),
            dict(
                method='L-BFGS-B',
                options={
                    'disp': verbose,
                    'maxiter': num_steps,
                },
            ))

        def closure():
            optimizer.zero_grad()
            loss = -self.evidence()
            loss.backward()
            return loss

        optimizer.step(closure)

    def _learn_with_adam(self, num_steps: int, verbose: bool = False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        progress_bar = tqdm(range(num_steps), disable=not verbose)
        for i in progress_bar:
            optimizer.zero_grad()
            loss = -self.evidence()
            loss.backward()
            optimizer.step()
            progress_bar.set_description(
                f"Iter: {i:02d} loss: {loss.item(): .2f}")

    @torch.no_grad()
    def predict(
        self,
        x_test: np.ndarray,
        diag_only: bool = True,
        include_likelihood: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        test_x = torch.tensor(x_test, self.dtype, self.device)
        mean, covar = self.forward(test_x, diag_only)
        if include_likelihood:
            mean, covar = self.likelihood(mean, covar, diag_only)
        return utils.to_array(mean), utils.to_array(covar)

    def common_terms(self):
        Kyy = self.kernel(self.train_x)
        Kyy.diagonal().add_(self.noise_variance)
        TrilKyy = utils.tril(Kyy)
        InvKyy_y = utils.tril_solve(TrilKyy, self.train_y)
        return TrilKyy, InvKyy_y

    def evidence(self) -> torch.Tensor:
        TrilKyy, InvKyy_y = self.common_terms()
        quadratic_term = (self.train_y * InvKyy_y).sum()
        log_term = utils.tril_logdet(TrilKyy)
        constant_term = len(self.train_y) * np.log(2 * np.pi)
        return -0.5 * (quadratic_term + log_term + constant_term)

    def forward(
        self,
        test_x: torch.Tensor,
        diag_only: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        TrilKyy, InvKyy_y = self.common_terms()
        Ksf = self.kernel(test_x, self.train_x)
        InvTrilKyy_Kfs = utils.tril_inv_matmul(TrilKyy, Ksf.T)
        mean = Ksf @ InvKyy_y
        if diag_only:
            kss = self.kernel.diag(test_x)
            covar = kss - utils.tril_diag_quadratic(InvTrilKyy_Kfs)
        else:
            Kss = self.kernel(test_x)
            covar = Kss - InvTrilKyy_Kfs.T @ InvTrilKyy_Kfs
        return mean, covar

    @property
    def noise_variance(self) -> torch.Tensor:
        return utils.free_to_constrained(self.free_noise_variance)

    @noise_variance.setter
    def noise_variance(self, value):
        self.free_noise_variance = Parameter(
            utils.constrained_to_free(
                torch.tensor(value, self.dtype, self.device)))
