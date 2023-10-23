from typing import List

import torch

from .base_kernel import BaseKernel


class AttentiveKernel(BaseKernel):
    def __init__(
        self,
        input_scales: List[float],
        output_scale: float,
        dim_input: int,
        dim_hidden: int,
        device_name: str = "cpu",
    ) -> None:
        super().__init__(output_scale, device_name)
        self.input_scales = self._to_tensor(input_scales)
        self.nn = MLP(
            dim_input,
            dim_hidden,
            len(input_scales),
        ).to(self.input_scales)

    @property
    def num_input_scales(self):
        return len(self.input_scales)

    def base_kernel(self, dist, input_scale):
        return dist.div(input_scale).square().mul(-0.5).exp()

    def embedding(self, x):
        z = self.nn(x)
        z_normed = z / z.norm(dim=1, keepdim=True)
        return z_normed

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        dist = torch.cdist(x1, x2, p=2)
        z1 = self.embedding(x1)
        z2 = self.embedding(x2)
        cov = 0.0
        for i in range(self.num_input_scales):
            similarity = torch.outer(z1[:, i], z2[:, i])
            cov += similarity * self.base_kernel(dist, self.input_scales[i])
        visibility = z1 @ z2.t()
        cov *= visibility
        return cov


class MLP(torch.nn.Sequential):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        self.add_module("linear1", torch.nn.Linear(dim_input, dim_hidden))
        self.add_module("activation1", torch.nn.Tanh())
        self.add_module("linear2", torch.nn.Linear(dim_hidden, dim_hidden))
        self.add_module("activation2", torch.nn.Tanh())
        self.add_module("linear3", torch.nn.Linear(dim_hidden, dim_output))
        self.add_module("activation3", torch.nn.Softmax(dim=1))
