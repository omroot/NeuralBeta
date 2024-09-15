import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple


class NeuralBetaEstimator(nn.Module):
    """
    A neural network model for estimating beta values with a GRU component.
    The network processes input features `x` and target values `y` through a GRU,
    and outputs a single beta value.

    Args:
        input_dim (int): The dimensionality of the input features `x`.
        hidden_dim (int): The number of units in the GRU hidden layer.
        gru_layers (int): The number of GRU layers.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 gru_layers: int = 1):
        super(NeuralBetaEstimator, self).__init__()

        # Initialize GRU with the correct input size
        self.gru = nn.GRU(input_dim+1, hidden_dim, num_layers=gru_layers, batch_first=True)

        # Fully connected layers to map GRU output to beta values
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # Hidden to hidden
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Hidden to output (single beta)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input features of shape (batch_size, seq_len, input_dim).
            y (torch.Tensor): Target values of shape (batch_size, seq_len, 1).

        Returns:
            torch.Tensor: Estimated beta values of shape (batch_size, 1).
        """
        # Concatenate x and y along the feature dimension
        combined_input = torch.cat((x, y), dim=-1)

        # Forward pass through the GRU
        # The combined input should have shape (batch_size, seq_len, input_dim)
        gru_out, _ = self.gru(combined_input)

        # Use the output of the last GRU step for the final prediction
        last_hidden = gru_out#[:, -1, :]

        # Forward pass through the fully connected layers
        out = torch.relu(self.fc1(last_hidden))
        out = self.fc2(out)
        beta = self.fc3(out)
        return beta


def train_beta_estimator(model: nn.Module,
                         train_loader: DataLoader,
                         optimizer: optim.Optimizer,
                         criterion: nn.Module) -> float:
    """
    Train the beta estimator model.

    Args:
        model (nn.Module): The beta estimator model.
        train_loader (DataLoader): DataLoader providing training data.
        optimizer (optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function used for training.

    Returns:
        float: The average loss over the training data.
    """
    model.train()
    total_loss = 0.0
    for x_t, y_t, x_t1, y_t1, true_beta_t, true_beta_t1 in train_loader:
        optimizer.zero_grad()

        # Forward pass: Get beta estimates
        beta = model(x_t, y_t)
        # Predict y_hat for next step
        y_hat = beta * x_t1

        # Calculate loss
        loss = criterion(y_hat, y_t1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_beta_estimator(model, test_loader, criterion):
    model.eval()
    estimated_betas = []
    true_betas = []
    predictions = []
    targets = []

    with torch.no_grad():
        for x_t, y_t, x_t1, y_t1, true_beta_t, true_beta_t1 in test_loader:
            # Estimate beta from the model
            beta = model(x_t, y_t)

            # Store the estimated and true betas
            estimated_betas.append(beta)
            true_betas.append(true_beta_t1)

            # Predict y_hat for the next step
            y_hat = beta * x_t1

            # Store predictions and targets
            predictions.append(y_hat)
            targets.append(y_t1)

        # Concatenate results
        estimated_betas = torch.cat(estimated_betas)
        true_betas = torch.cat(true_betas)
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        # Calculate beta RMSE
        beta_rmse = np.sqrt(((estimated_betas - true_betas) ** 2).mean().item())

        # Calculate prediction RMSE
        mse_loss = criterion(predictions, targets)
        prediction_rmse = np.sqrt(mse_loss.item())

    return prediction_rmse, beta_rmse, true_betas, estimated_betas


def estimate_ols_beta(x_train, y_train):
    """
    Estimate betas using Ordinary Least Squares (OLS).

    Parameters:
    - x_train (numpy.ndarray): Input data of shape (sample_size, seq_len, input_dim).
    - y_train (numpy.ndarray): Target data of shape (sample_size, seq_len, input_dim).

    Returns:
    - beta_ols (numpy.ndarray): Estimated beta values of shape (sample_size, seq_len, input_dim).
    """
    sample_size, seq_len, input_dim = x_train.shape

    beta_ols = np.zeros_like(x_train)  # Initialize beta estimates

    for i in range(sample_size):
        for t in range(seq_len):
            X = x_train[i, t, :]
            Y = y_train[i, t, :]
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            # Add a bias term if needed
            X_bias = np.hstack((X, np.ones((X.shape[0], 1))))
            beta = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ Y
            beta_ols[i, t, :] = beta[:-1]  # Exclude bias term from the result

    return beta_ols