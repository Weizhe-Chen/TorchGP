import torch
from torch.nn.parameter import Parameter

from .. import utils
from .base_kernel import BaseKernel


class GaussianKernel(BaseKernel):
    def __init__(
        self,
        input_scale: float,
        output_scale: float,
        device_name: str = "cpu",
    ) -> None:
        super().__init__(output_scale, device_name)
        self.input_scale = input_scale

    @property
    def input_scale(self) -> torch.Tensor:
        return utils.free_to_constrained(self.free_input_scale)

    @input_scale.setter
    def input_scale(self, value):
        self.free_input_scale = Parameter(
            utils.constrained_to_free(
                torch.tensor(value, dtype=self.dtype, device=self.device)
            )
        )

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        x1 = x1 / self.input_scale
        x2 = x2 / self.input_scale
        dist = torch.cdist(x1, x2, p=2)
        return self.output_scale * torch.exp(-0.5 * dist.square())
