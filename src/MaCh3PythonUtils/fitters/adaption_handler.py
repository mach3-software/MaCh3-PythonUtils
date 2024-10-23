import numpy as np
from numba import njit

@njit
def update_covariance(mean, covariance, new_data, count):
    """Update the mean and covariance matrix efficiently."""
    delta = new_data - mean
    mean += delta / count
    
    delta2 = new_data - mean
    covariance += np.outer(delta, delta2) * (count - 1) / count
    
    return mean, covariance


class CovarianceUpdater:
    def __init__(self, n_dimensions, update_step=1000):
        self.n = n_dimensions
        self.mean = np.zeros(n_dimensions)
        self.covariance = np.zeros((n_dimensions, n_dimensions))
        self.frozen_covariance = np.identity(n_dimensions)
        
        self.count = 0
        self.update_step=update_step
        self._rng = np.random.default_rng()


    def update(self, new_data):
        """Update the mean and covariance with a new data point.

        Parameters:
        new_data (np.ndarray): A 1D array representing the new data point (shape: (n_dimensions,))
        """
        # if new_data.shape[0] != self.n:
        #     raise ValueError(f"Expected data with {self.n} dimensions, got {new_data.shape[0]}.")

        self.count += 1
        
        # Update mean and covariance using the Numba-optimized function
        self.mean, self.covariance = update_covariance(self.mean, self.covariance, new_data, self.count)

        if self.count%self.update_step and self.count>100:
            self.frozen_covariance = self.get_covariance().copy()

    def get_mean(self):
        """Return the current mean."""
        return self.mean

    def get_covariance(self):
        """Return the current covariance matrix."""
        return self.covariance / (self.count - 1) if self.count > 1 else np.zeros((self.n, self.n))

    def get_frozen_covariance(self):
        return (2.4**2)/(float(self.n)) * self.frozen_covariance + np.identity(self.n)*1e-6

    def sample(self, n_samples: int):
        """Sample from a multivariate Gaussian centered at 'center'.

        Parameters:
        center (np.ndarray): A 1D array representing the center for the Gaussian distribution (shape: (n_dimensions,))
        num_samples (int): The number of samples to draw.

        Returns:
        np.ndarray: Samples drawn from the multivariate Gaussian distribution.
        """
        return self._rng.multivariate_normal(mean=np.zeros(self.n), size=n_samples,
                                             cov=self.get_frozen_covariance(), tol=1e-6)

