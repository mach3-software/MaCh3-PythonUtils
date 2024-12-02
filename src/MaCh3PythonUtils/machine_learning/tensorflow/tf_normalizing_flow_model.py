from MaCh3PythonUtils.machine_learning.tensorflow.tf_manual_interface import TfManualInterface

import tensorflow_probability as tfp
import tensorflow.keras as tfk
import tensorflow as tf
from typing import List
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

tfd = tfp.distributions
tfb = tfp.bijectors


# class FlowLayer(tfk.layers.Layer):
#     def __init__(self, dist):
#         super().__init__()
#         self._layer_dist = dist.log_prob
    
#     def call(self, x):
#         return self._layer_dist(x)


class NormalizingFlow:
    def __init__(self, hidden_units: List[int], n_bijectors: int, input_dim: int):
        self._hidden_units = hidden_units
        self._n_bijectors = n_bijectors
        self._input_dim=input_dim
        
    # Create an autoregressive bijector
    def _create_autoregressive_bijector(self):
        return tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                params=2,  # Shift and log scale
                hidden_units=self._hidden_units,
                input_shape=(None,self._input_dim),
            )
        )
        
    def _create_normalizing_flow(self):
        bijectors = [self._create_autoregressive_bijector() for _ in range(self._n_bijectors)]
        # Add a final bijector for numerical stability (optional)
        bijectors.append(tfb.Permute(permutation=list(range(self._input_dim))[::-1]))
        return tfb.Chain(bijectors)

    def __call__(self):
        flow_bijector = self._create_normalizing_flow()
        base_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros(self._input_dim), scale_diag=tf.ones(self._input_dim))
        # Grab the distribution
        return tfd.TransformedDistribution(distribution=base_distribution, bijector=flow_bijector)




class TfNormalizingFlowModel(TfManualInterface):
    def build_model(self, model_args):
        input_dim = self.chain.ndim-1
        self._model = NormalizingFlow(model_args.get("hidden_units", [100]), model_args.get("n_bijectors", 1), input_dim)()
        self._optimizer = tfk.optimizers.Adam(model_args.get("learning_rate", 1e-3))

    def nll_loss(self, features):
        return -tf.reduce_mean(self._model.log_prob(features))


    def train_model(self):
        epochs = self._training_settings["epochs"]
        batch_size = self._training_settings["batch_size"]
        
        scaled_data = tf.data.Dataset.from_tensor_slices(self.scale_data(self._training_data)).batch(batch_size)

        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            for batch in scaled_data:
                with tf.GradientTape() as tape:
                    loss = self.nll_loss(batch)
                grads = tape.gradient(loss, self._model.trainable_variables)
                self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
                epoch_loss += loss.numpy()
            print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(scaled_data)}")

    def test_model(self):
        print("Testing")
        surrogate_samples = self._model.sample(10000).numpy()

        with PdfPages("likelihood_free_inference.pdf") as pdf:
            for i in range(self._chain.ndim-1):
                sns.histplot(surrogate_samples[:, i], label="Surrogate", color="blue", fill=False)
                sns.histplot(self._chain.ttree_array[:, i], label="Test Data", color="blue", fill=False)
                plt.legend()
                plt.xlabel(f"{self._chain.plot_branches[i]}")
                plt.ylabel("Density")
                pdf.savefig()
                plt.close()
