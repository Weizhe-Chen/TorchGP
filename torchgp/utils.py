from typing import Tuple

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
