---
title: "Gaussian Process Regression"
---

``` {=html}
<style>
body { min-width: 80% !important; }
</style>
```

## Prediction

$$
\begin{aligned}
p(\mathbf{f}_{\star}|\mathbf{y}) &= \mathcal{N}(\mathbf{f}_{\star}|\boldsymbol{\mu},\boldsymbol{\Sigma}),\\
\boldsymbol{\mu} &= \mathbf{K}_{\star x}\mathbf{K}_{y}^{-1}\mathbf{y},\\
\boldsymbol{\Sigma} &= \mathbf{K}_{\star \star} - \mathbf{K}_{\star x}\mathbf{K}_{y}^{-1}\mathbf{K}_{x \star},\\
\text{where }\mathbf{K}_{y}^{-1} &= \mathbf{K}_{x x} + \sigma^{2} \mathbf{I}_{N}
\end{aligned}
$$

## Model Evidence

$$
\ln p(\mathbf{y}|\boldsymbol{\theta}) = -\frac{1}{2}\left(\mathbf{y}^{\top}\mathbf{K}_{y}^{-1}\mathbf{y} + \ln|\mathbf{K}_{y}| + N\ln(2\pi)\right)
$$

## Practical Implementation

```python
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.parameter import Parameter
from tqdm import tqdm
```

### Parameter Transformations

We use the `softplus-and-shift` transformations to ensure that the kernel parameters are positive:

```python
def free_to_positive(free_param):
    return F.softplus(free_param, 1.0, 20.0) + 1e-6


def positive_to_free(positive_param):
    if torch.any(positive_param <= 1e-6):
        raise ValueError("Input to `positive_to_free` must be greater than 1e-6.")
    shifted_positive_param = positive_param - 1e-6
    return shifted_positive_param + torch.log(-torch.expm1(-shifted_positive_param))
```

### Tensor Utilities

Typically, we will want to convert between NumPy arrays and PyTorch tensors.
We set the default tensor type to `torch.float64` to allow for higher precision calculations.
This is important for the Cholesky decomposition, which can be numerically unstable for low precision in some models.

```python
def to_tensor(input):
    return torch.tensor(input, dtype=torch.float64)


def tensor_to_array(tensor):
    return tensor.detach().cpu().numpy()
```

### Robust Cholesky Decomposition

Cholesky decomposition fails when the input matrix is not positive-definite.
Theoretically this should never happen, but in practice it can happen due to numerical errors.
We can add a small amount of positive value, also known as `jitter`, to the diagonal of the matrix to make it positive-definite.
This is a common technique in the implementation of Gaussian process models.
Sometimes, even after adding jitter, the matrix is still not positive-definite.
In this case, we can try adding more jitter and attempting the decomposition again.
We can repeat this process until the decomposition succeeds or until we reach a maximum number of attempts.

```python
def robust_cholesky(A, base_jitter=1e-6, num_attempts=3):
    # cholesky_ex skips the slow error checking and error message construction
    # of `torch.linalg.cholesky()`. Returns the LAPACK error codes as part of
    # a named tuple (L, info). This makes this function a faster way to check
    # if a matrix is positive-definite, and it provides a way to handle
    # decomposition errors more gracefully or performantly.
    # If `A` is not a Hermitian positive-definite matrix, then `info` will be
    # positive integers, indicating the order of the leading minor that is not
    # positive-definite. If `info` is filled with zeros, then the decomposition
    # was successful and `L` is the lower Cholesky factor.
    L, info = torch.linalg.cholesky_ex(A)
    if not torch.any(info):
        # The decomposition was successful.
        return L
    if torch.isnan(A).any():
        raise ValueError("Input to `robust_cholesky` must not contain NaNs.")
    _A = A.clone()
    prev_jitter = 0.0
    jitter = base_jitter
    for i in range(num_attempts):
        not_positive_definite = info > 0
        jitter = base_jitter * (10**i)
        increment = not_positive_definite * (jitter - prev_jitter)
        _A.diagonal().add_(increment)
        prev_jitter = jitter
        warnings.warn(
            f"Added jitter of {jitter} to the diagonal of a matrix that was not"
            + "positive-definite. Attempting Cholesky decomposition again."
        )
        L, info = torch.linalg.cholesky_ex(_A)
        if not torch.any(info):
            return L
    raise ValueError(
        f"Cholesky decomposition failed after adding {jitter} to the diagonal."
        + "Try increasing the `base_jitter` or `num_attempts` arguments."
    )
```

### Gaussian Process Regression

We use the following numerical tricks to make the implementation more efficient and numerically stable:

- $\texttt{diag}(\mathbf{K}_{\star x}\mathbf{K}_{y}^{-1}\mathbf{K}_{x \star})$ is computed via `iLyy_Ksx.pow(2).sum(dim=0).view(-1, 1)`.

- Log determinant of $\mathbf{K}_{y}$ is computed via `Lyy.diagonal().square().log().sum()` using the identity $\ln|\mathbf{K}_{y}| = \sum_{i}\log\ell_{ii}^{2}$, where $\ell_{ii}$ is the $i$-th diagonal element of the Cholesky factor $\mathbf{L}_{y}$.

```python
class GPR(torch.nn.Module):
    def __init__(self, x_train, y_train, kernel, noise_variance):
        super().__init__()
        self.set_data(x_train, y_train)
        self.kernel = kernel
        self.__free_noise_variance = Parameter(
            positive_to_free(to_tensor(noise_variance))
        )

    def set_data(self, x_train, y_train):
        self.train_x = to_tensor(x_train)
        self.train_y = to_tensor(y_train)

    @property
    def noise_variance(self):
        var = free_to_positive(self.__free_noise_variance)
        return var

    @torch.no_grad()
    def predict(self, test_x, diag=True, include_noise=True):
        test_x = to_tensor(test_x)
        mean, variance = self.forward(test_x, diag=diag)
        if include_noise:
            variance += self.noise_variance
        return tensor_to_array(mean), tensor_to_array(variance)

    def fit(self, num_steps, lr=0.01, verbose=True):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        progress = range(num_steps)
        if verbose:
            progress = tqdm(progress)
        for i in progress:
            optimizer.zero_grad()
            loss = -self.model_evidence()
            loss.backward()
            if verbose:
                progress.set_description(f"Iter: {i:04d} " + f"Loss: {loss.item():.4f}")
        self.eval()

    def forward(self, test_x, diag=True):
        Kyy = self.kernel(self.train_x)
        Kyy.diagonal().add_(self.noise_variance)
        Lyy = robust_cholesky(Kyy, base_jitter=0.0)
        Ksx = self.kernel(test_x, self.train_x)
        iLyy_Ksx = torch.linalg.solve_triangular(Lyy, Ksx.T, upper=False)
        alpha = torch.cholesky_solve(self.train_y, Lyy)  # iKyy_y
        mean = Ksx @ alpha
        if diag:
            kss = self.kernel(test_x, diag=True)
            variance = kss - iLyy_Ksx.pow(2).sum(dim=0).view(-1, 1)
        else:
            Kss = self.kernel(test_x, test_x)
            variance = Kss - iLyy_Ksx.T @ iLyy_Ksx
        return mean, variance

    def model_evidence(self):
        Kyy = self.kernel(self.train_x)
        Kyy.diagonal().add_(self.noise_variance)
        Lyy = robust_cholesky(Kyy, base_jitter=0.0)
        alpha = torch.cholesky_solve(self.train_y, Lyy)
        num_data = self.train_y.shape[0]
        quadratic = -0.5 * torch.sum(alpha * self.train_y)
        log_det = -Lyy.diagonal().square().log().sum()
        constant = -0.5 * num_data * np.log(2 * np.pi)
        return quadratic + log_det + constant
```

### A Simple Example

```python
class GaussianKernel(torch.nn.Module):
    def __init__(self, length_scale, output_scale):
        super().__init__()
        self.__free_length_scale = Parameter(positive_to_free(to_tensor(length_scale)))
        self.__free_output_scale = Parameter(positive_to_free(to_tensor(output_scale)))

    @property
    def length_scale(self):
        return free_to_positive(self.__free_length_scale)

    @property
    def output_scale(self):
        return free_to_positive(self.__free_output_scale)

    def forward(self, x1, x2=None, diag=False):
        if diag:
            return self.output_scale * torch.ones(x1.shape[0], 1).to(x1)
        if x2 is None:
            x2 = x1
        x1 = x1 / self.length_scale
        x2 = x2 / self.length_scale
        dist = torch.cdist(x1, x2, p=2)
        return self.output_scale * torch.exp(-0.5 * dist.pow(2))


def main():
    # Generate data.
    np.random.seed(0)
    x_train = np.random.uniform(-5.0, 5.0, size=(100, 1))
    y_train = np.sin(x_train) + np.random.normal(0.0, 0.1, size=(100, 1))
    x_test = np.linspace(-5.0, 5.0, num=100).reshape(-1, 1)

    # Fit a Gaussian process.
    kernel = GaussianKernel(length_scale=1.0, output_scale=1.0)
    gpr = GPR(x_train, y_train, kernel, noise_variance=0.01)
    gpr.fit(num_steps=200, lr=0.01)

    # Plot the results.
    mean, variance = gpr.predict(x_test)
    std = np.sqrt(variance)

    _, ax = plt.subplots()
    ax.plot(x_train, y_train, "kx")
    ax.plot(x_test, mean, "C0")
    ax.fill_between(
        x_test.flatten(),
        mean.flatten() - 2.0 * std.flatten(),
        mean.flatten() + 2.0 * std.flatten(),
        color="C0",
        alpha=0.2,
    )
    ax.set_xlim(-5.0, 5.0)
    plt.show()


if __name__ == "__main__":
    main()
```

Running this example produces the following plot

![](../images/gpr.svg)
