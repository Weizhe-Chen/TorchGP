from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def free_to_constrained(x, min_value=1e-6):
    return F.softplus(x) + min_value


def constrained_to_free(x, min_value=1e-6):
    if torch.any(x <= min_value):
        raise ValueError(f"Input must be greater than {min_value}.")
    shifted_x = x - min_value
    return shifted_x + torch.log(-torch.expm1(-shifted_x))


def get_dtype_and_device(device_name: str) -> Tuple[torch.dtype, torch.device]:
    """
    Returns the PyTorch dtype and device associated with the given device name.

    Parameters
    ----------
    device_name : str
        The name of the device. Must be one of 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[torch.dtype, torch.device]
        A tuple containing the dtype and device.

    Raises
    ------
    ValueError
        If the device name is not 'cpu' and does not contain the string 'cuda'.
    """
    if device_name == "cpu":
        dtype = torch.double
    elif "cuda" in device_name:
        dtype = torch.float
    else:
        raise ValueError("Invalid device name.")
    device = torch.device(device_name)
    return dtype, device


def to_array(tensor):
    return tensor.detach().cpu().numpy()


def tril(A, base_jitter=1e-6, num_attempts=3):
    L, info = torch.linalg.cholesky_ex(A)
    if not torch.any(info):
        return L  # The decomposition was successful.
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
        L, info = torch.linalg.cholesky_ex(_A)
        if not torch.any(info):
            return L
    raise ValueError(
        f"Cholesky decomposition failed after adding {jitter} to the diagonal."
        + "Try increasing the `base_jitter` or `num_attempts` arguments."
    )


def tril_solve(L, B):
    """
    Returns A^{-1} @ B, where A = L @ L.T and L is a lower-triangular matrix.
    """
    return torch.cholesky_solve(B, L)


def tril_inv_matmul(L, B):
    """
    Returns L^{-1} @ B, where L is a lower-triangular matrix.
    """
    return torch.linalg.solve_triangular(L, B, upper=False)


def tril_diag_quadratic(L):
    """
    returns a column vector of the diagonal elements of L^{T} @ L

    """
    return L.square().sum(dim=0).view(-1, 1)


def tril_logdet(L):
    """
    Computes the log determinant of a matrix A = L @ L.T using the
    Cholesky decomposition of A.
    """
    return 2.0 * L.diagonal().log().sum()


def trace_quadratic(A):
    """
    returns the trace of A.T @ A or A @ A.T
    """
    return A.square().sum()


class StandardScaler:
    def __init__(self, values):
        if values.ndim != 2:
            raise ValueError("values.shape=(num_samples, num_dims)")
        self.scale = values.std(axis=0, keepdims=True)
        if np.any(self.scale <= 0.0):
            raise ValueError("scale must be positive")
        self.mean = values.mean(axis=0, keepdims=True)

    def preprocess(self, raw):
        transformed = (raw - self.mean) / self.scale
        return transformed

    def postprocess_mean(self, transformed):
        raw = transformed * self.scale + self.mean
        return raw

    def postprocess_std(self, transformed):
        raw = transformed * self.scale
        return raw

    def postprocess_covar(self, transformed):
        raw = transformed * (self.scale ** 2)
        return raw


class MinMaxScaler:
    def __init__(self, values, expected_range=(-1.0, 1.0)):
        self.min = expected_range[0]
        self.max = expected_range[1]
        # `ptp` is the acronym for ‘peak to peak’.
        self.ptp = expected_range[1] - expected_range[0]
        if self.ptp <= 0.0:
            raise ValueError("Expected range must be positive.")
        self.data_min = values.min(axis=0, keepdims=True)
        self.data_max = values.max(axis=0, keepdims=True)
        self.data_ptp = self.data_max - self.data_min
        if np.any(self.data_ptp <= 0.0):
            raise ValueError("Data range must be positive.")

    def preprocess(self, raw):
        standardized = (raw - self.data_min) / self.data_ptp
        transformed = standardized * self.ptp + self.min
        return transformed

    def postprocess(self, transformed):
        standardized = (transformed - self.min) / self.ptp
        raw = standardized * self.data_ptp + self.data_min
        return raw
