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
