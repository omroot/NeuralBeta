import numpy as np

def compute_rmse(estimated_betas, true_betas):
    """
    Compute the Root Mean Squared Error (RMSE) between estimated and true betas.

    Parameters:
    - estimated_betas (numpy.ndarray): Estimated beta values of shape (sample_size, seq_len, input_dim).
    - true_betas (numpy.ndarray): True beta values of shape (sample_size, seq_len, input_dim).

    Returns:
    - rmse (float): The computed RMSE value.
    """
    mse = np.mean((estimated_betas - true_betas) ** 2)
    rmse = np.sqrt(mse)
    return rmse