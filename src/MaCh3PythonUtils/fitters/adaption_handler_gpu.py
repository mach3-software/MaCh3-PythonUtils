import tensorflow as tf

class CovarianceUpdaterGPU:
    def __init__(self, n_dimensions, update_step=1000):
        self.n = n_dimensions
        # Initialize mean and covariance as TensorFlow tensors
        self.mean = tf.Variable(tf.zeros(n_dimensions, dtype=tf.float32))
        self.covariance = tf.Variable(tf.zeros((n_dimensions, n_dimensions), dtype=tf.float32))
        self.frozen_covariance = tf.eye(n_dimensions, dtype=tf.float32)
        
        self.count = 0
        self.update_step = update_step

    @tf.function
    def update_covariance(self, new_data):
        """Update the mean and covariance matrix efficiently using TensorFlow."""
        count = self.count + 1
        delta = new_data - self.mean
        self.mean.assign_add(delta / count)
        delta2 = new_data - self.mean
        self.covariance.assign_add(tf.linalg.matmul(tf.expand_dims(delta, axis=-1), 
                                                     tf.expand_dims(delta2, axis=0)) * (count - 1) / count)
        return self.mean, self.covariance

    @tf.function
    def update(self, new_data):
        """Update the mean and covariance with a new data point.

        Parameters:
        new_data (tf.Tensor): A 1D tensor representing the new data point (shape: (n_dimensions,))
        """
        self.count += 1
        
        # Arbitary stopping point!
        if self.count>100000:
            return
        
        # Update mean and covariance using the class method
        self.mean, self.covariance = self.update_covariance(new_data)

        if self.count % self.update_step == 0 and self.count > 100:
            self.frozen_covariance.assign(self.get_covariance())

    @tf.function
    def get_mean(self):
        """Return the current mean."""
        return self.mean

    @tf.function
    def get_covariance(self):
        """Return the current covariance matrix."""
        return self.covariance / (self.count - 1) if self.count > 1 else tf.eye((self.n), dtype=tf.float32)

    def get_frozen_covariance(self):
        return (2.4 ** 2) / float(self.n) * self.frozen_covariance + tf.eye(self.n) * 1e-6

    @tf.function
    def sample(self, n_samples: int):
        """Sample from a multivariate Gaussian centered at 0.

        Parameters:
        n_samples (int): The number of samples to draw.

        Returns:
        tf.Tensor: Samples drawn from the multivariate Gaussian distribution.
        """
        frozen_cov = self.get_frozen_covariance()
        mvn = tf.random.normal(shape=(n_samples, self.n), mean=0.0, stddev=1.0)
        chol = tf.linalg.cholesky(frozen_cov)
        samples = tf.linalg.matmul(mvn, chol)  # Centering the samples at the mean
        return samples
