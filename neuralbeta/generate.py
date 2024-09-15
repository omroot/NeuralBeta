
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Any
from sklearn.metrics import mean_squared_error


class SyntheticDataGenerator:
    """
    A class to generate synthetic time series data based on different types of beta patterns.
    
    Attributes:
        length (int): The length of each time series (number of time steps).
        n_samples (int): The number of samples (time series) to generate.
        beta_type (str): The type of beta pattern used to generate data.
    
    Methods:
        generate_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Generates and returns the synthetic data based on the configured beta type.
    """
    
    def __init__(self, length: int = 65, n_samples: int = 100000, beta_type: str = 'constant') -> None:
        """
        Initializes the SyntheticDataGenerator with the provided configuration.
        
        Args:
            length (int): The length of each time series (number of time steps). Default is 65.
            n_samples (int): The number of samples (time series) to generate. Default is 100,000.
            beta_type (str): The type of beta pattern to generate ('constant', 'stepwise', or 'cyclical'). Default is 'constant'.
        
        Raises:
            ValueError: If an unsupported beta_type is provided.
        """
        self.length = length
        self.n_samples = n_samples
        self.beta_type = beta_type

        # Validate beta_type
        if beta_type not in ['constant', 'stepwise', 'cyclical']:
            raise ValueError("Unsupported beta_type. Use 'constant', 'stepwise' or 'cyclical'.")
    
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates synthetic time series data based on the configured beta type.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - x (np.ndarray): The generated independent variable (predictor) data of shape (n_samples, length).
                - y (np.ndarray): The generated dependent variable (response) data of shape (n_samples, length).
                - beta_t (np.ndarray): The time-varying beta coefficients used to generate the response data.
        """
        
        # Generate independent variable data (x) and noise (epsilon)
        x = np.random.standard_t(df=10, size=(self.n_samples, self.length))
        epsilon = np.random.normal(0, 1, size=(self.n_samples, self.length))
        
        # Generate beta coefficients based on the beta_type
        if self.beta_type == 'constant':
            beta = np.random.normal(1, 1, size=self.n_samples)
            beta_t = np.repeat(beta[:, np.newaxis], self.length, axis=1)
        
        elif self.beta_type == 'stepwise':
            beta = np.concatenate([np.random.normal(1, 1, size=(self.n_samples // 2, self.length)),
                                    np.random.normal(1, 1, size=(self.n_samples // 2, self.length))], axis=0)
            beta_t = np.concatenate([np.ones((self.n_samples // 2, self.length)) * beta[:self.n_samples // 2, 0].reshape(-1, 1),
                                      np.ones((self.n_samples // 2, self.length)) * beta[self.n_samples // 2:, 0].reshape(-1, 1)], axis=0)

        
        elif self.beta_type == 'cyclical':
            beta0 = np.random.normal(0, 1, size=self.n_samples)
            c = np.random.uniform(4, 32, size=self.n_samples)
            t = np.arange(self.length)
            beta_t = np.sin(beta0[:, np.newaxis] + c[:, np.newaxis] * t)
        
        # Generate dependent variable data (y)
        y = beta_t * x + epsilon
        
        return x, y, beta_t
def generate_synthetic_data(x, y, true_beta, num_samples, sequence_length):
    """
    Generate synthetic data including true betas for time steps t and t+1.

    Parameters:
    - x (torch.Tensor): Input data of shape (num_samples, sequence_length, features).
    - y (torch.Tensor): Target data of shape (num_samples, sequence_length, targets).
    - true_beta (torch.Tensor): True beta values of shape (num_samples, sequence_length, features).
    - num_samples (int): Number of samples in the dataset.
    - sequence_length (int): Length of the sequence (number of time steps).

    Returns:
    - x_t (torch.Tensor): Data up to time step t.
    - y_t (torch.Tensor): Data up to time step t.
    - x_t1 (torch.Tensor): Data for time step t+1.
    - y_t1 (torch.Tensor): Data for time step t+1.
    - true_beta_t (torch.Tensor): True beta values for time step t.
    - true_beta_t1 (torch.Tensor): True beta values for time step t+1.
    """
    # Data for time step t
    x_t = x[:, :-1, :]  # Data up to time step t
    y_t = y[:, :-1, :]  # Data up to time step t

    # Data for time step t+1
    x_t1 = x[:, 1:, :]  # Data for time step t+1
    y_t1 = y[:, 1:, :]  # Data for time step t+1

    # Extract true beta values for time steps t and t+1
    true_beta_t = true_beta[:, :-1, :]  # True beta values for time step t
    true_beta_t1 = true_beta[:, 1:, :]  # True beta values for time step t+1

    return (
        torch.tensor(x_t, dtype=torch.float32),
        torch.tensor(y_t, dtype=torch.float32),
        torch.tensor(x_t1, dtype=torch.float32),
        torch.tensor(y_t1, dtype=torch.float32),
        torch.tensor(true_beta_t, dtype=torch.float32),
        torch.tensor(true_beta_t1, dtype=torch.float32)
    )
