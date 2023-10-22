from abc import ABCMeta, abstractmethod
from typing import Tuple
import numpy as np
import torch
from .. import utils

from ..kernels import BaseKernel
from ..likelihoods import BaseLikelihood


class BaseModel(torch.nn.Module, metaclass=ABCMeta):
    """Interface of a model."""

    @abstractmethod
    def __init__(
        self,
        kernel: BaseKernel,
        likelihood: BaseLikelihood,
        device_name: str = "cpu",
    ) -> None:
        r"""Initializes a model with the specified device name.

        Args:
            device_name (str): The name of the PyTorch device to be used for
                computations.

        """
        super().__init__()
        self.dtype, self.device = utils.get_dtype_and_device(device_name)
        self.kernel = kernel
        self.likelihood = likelihood

    @abstractmethod
    def learn(
        self,
        x_new: np.ndarray,
        y_new: np.ndarray,
        optimizer: str,
        args: dict,
    ) -> None:
        r"""Optimizes the model parameters.

        Args:
            x_new (np.ndarray): New training inputs of shape
                (num_inputs, dim_inputs).
            y_new (np.ndarray): New training outputs of shape
                (num_outputs, dim_outputs).
            optimizer (str): Name of the optimizer to be used:
                * adam
                * l-bfgs-b
            args (dict): Parameters for the optimizer.

        Raises:
            NotImplementedError: This is an abstract method and must be
                implemented by derived classes.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Makes predictions.

        Args:
            x_test (np.ndarray): Test inputs of shape (num_inputs, dim_inputs).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing predictive mean
                and predictive standard deviation of shape (num_inputs, 1).

        Raises:
            NotImplementedError: This is an abstract method and must be
                implemented by derived classes.
        """
        raise NotImplementedError

    def add_data(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        self.validate_data(x_new, y_new)
        if not (hasattr(self, "x_train") and hasattr(self, "y_train")):
            self.train_x = self._to_tensor(x_new)
            self.train_y = self._to_tensor(y_new)
        else:
            new_x = self._to_tensor(x_new)
            new_y = self._to_tensor(y_new)
            self.x_train = torch.cat((self.x_train, new_x), dim=0)
            self.y_train = torch.cat((self.y_train, new_y), dim=0)

    def validate_data(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        r"""Check if the inputs `x_new` and `y_new` are valid.

        Args:
            x_new (np.ndarray): An array of shape (num_inputs, dim_inputs)
                containing the input features of the new data.
            y_new (np.ndarray): An array of shape (num_outputs, 1)
                containing the output targets of the new data.

        Raises:
            ValueError: If any of the following conditions are met:
                1. `x_new` is not 2D.
                2. `y_new` is not 2D.
                3. `y_new` has more than 1 column.
                4. `x_new` and `y_new` have different number of samples.
                5. `x_new` and `self.x_train` have different number of features.
                6. `y_new` and `self.y_train` have different number of columns.

        """
        if x_new.ndim != 2:
            raise ValueError("x_train must be 2D.")
        if y_new.ndim != 2:
            raise ValueError("y_train must be 2D.")
        if y_new.shape[1] != 1:
            raise ValueError("Only support univariate output for now.")
        if x_new.shape[0] != y_new.shape[0]:
            raise ValueError("x_train and y_train should have same length.")
        if hasattr(self, "x_train") and x_new.shape[1] != self.x_train.shape[1]:
            raise ValueError("x_train and x_new should have same shape.")
        if hasattr(self, "y_train") and y_new.shape[1] != self.y_train.shape[1]:
            raise ValueError("y_train and y_new should have same shape.")

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype, device=self.device)

    @property
    def num_train(self) -> int:
        if not hasattr(self, "train_x"):
            return 0
        return self.train_x.shape[0]
