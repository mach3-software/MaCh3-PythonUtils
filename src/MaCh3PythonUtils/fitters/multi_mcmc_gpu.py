import emcee
import numpy as np
import tensorflow as tf
from tensorflow import linalg as tfla
from numpy.typing import NDArray
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
from typing import List
from MaCh3PythonUtils.machine_learning.tf_interface import TfInterface
from MaCh3PythonUtils.fitters.adaption_handler_gpu import CovarianceUpdaterGPU

class MCMCMultGPU:
    def __init__(self, interface: TfInterface, n_chains: int = 1024, circular_params: List[str] = [], update_step: int = 10, start_matrix_throw: int = 0):
        print("MCMC let's go!")

        self._interface = interface        
        self._n_dim = interface.chain.ndim - 1
        self._n_chains = n_chains
        
        # Use TensorFlow to ensure operations are performed on the GPU
        self._upper_bounds = tf.convert_to_tensor(interface.chain.upper_bounds[:-1], dtype=tf.float32)       
        self._lower_bounds = tf.convert_to_tensor(interface.chain.lower_bounds[:-1], dtype=tf.float32)
        self._sampler = None

        # Initial states for all chains
        initial_state = tf.convert_to_tensor(interface.training_data.iloc[[1]].to_numpy()[0], dtype=tf.float32)
        # Instead of using tf.fill, use tf.tile to replicate the initial_state across chains
        self._chain_states = tf.Variable(tf.tile(tf.expand_dims(initial_state, axis=0), [n_chains, 1]), dtype=tf.float32)

        self._circular_indices = [self._interface.chain.plot_branches.index(par) for par in circular_params]
        self._matrix_scale = 2.4**2 / float(self._n_dim)
        self._start_matrix_throw = start_matrix_throw
        self._update_step = update_step

        # CovarianceUpdater will be updated based on the first chain
        self._matrix_handler = CovarianceUpdaterGPU(self._n_dim, update_step)
        
        # Calculate likelihoods for all chains (with GPU processing)
        self._current_loglikelihoods = tf.Variable(self._calc_likelihood(self._chain_states))
        self._current_step = 0

    def _wrap_circular(self, state):
        """Apply circular boundary conditions."""
        return (state + np.pi) % (2 * np.pi) - np.pi

    @tf.function
    def _calc_likelihood(self, states: tf.Tensor):
        return -1 * self._interface.model_predict_no_scale(states)

    @tf.function(reduce_retracing=True)
    def propose_step_gpu(self):
        # Propose new states for all chains
        proposed_states = self._matrix_handler.sample(self._n_chains) + self._chain_states

        # Calculate log-likelihoods for proposed states
        proposed_loglikelihoods = self._calc_likelihood(proposed_states)

        # Metropolis-Hastings acceptance step
        log_diff = proposed_loglikelihoods - self._current_loglikelihoods
        acc_probs = tf.minimum(1.0, tf.exp(tf.clip_by_value(log_diff, -100, 0)))

        # Generate uniform random values for comparison
        u = tf.random.uniform(shape=(self._n_chains,1), dtype=tf.float32)

        # Determine which states to accept (True or False for each chain)
        accept = acc_probs > u

        # Use `tf.where` to conditionally update the chain states based on acceptance
        self._chain_states.assign(tf.where(accept, proposed_states, self._chain_states))

        self._current_loglikelihoods.assign(tf.where(accept, proposed_loglikelihoods, self._current_loglikelihoods))

    def propose_step(self):
        self.propose_step_gpu()
        # Update the covariance matrix using the first chain only
        self._matrix_handler.update(self._chain_states.numpy()[0])

        # Save the current state of all chains
        self._dataset[self._current_step, :] = self._chain_states.numpy()
        self._current_step += 1

    def save_mcmc_chain_to_pdf(self, filename: str, output_pdf: str):
        # Open the HDF5 file
        with h5py.File(filename, 'r') as f:
            chain = f['chain'][:]

        _, n_params = chain.shape[1:]
        
        # Create a PdfPages object to save plots
        with PdfPages(output_pdf) as pdf:
            for i in tqdm(range(n_params)):
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot the chain for the i-th parameter
                # for j in range(self._n_chains):
                ax.plot(chain[:, 0, i], lw=0.5, label=f'Chain {j+1}')
                ax.set_ylabel(self._interface.chain.plot_branches[i])
                ax.set_title(f"Parameter {self._interface.chain.plot_branches[i]} MCMC Chain")
                ax.set_xlabel('Step')
                # ax.legend()

                # Save the current figure to the PDF
                pdf.savefig(fig)
                plt.close(fig)  # Close the figure to save memory

        print(f"MCMC chain plots saved to {output_pdf}")

    def __call__(self, n_steps, output_file_name: str):
        print(f"Running MCMC for {n_steps} steps with {self._n_chains} chains")

        # Open the HDF5 file and create the dataset
        with h5py.File(output_file_name, 'w') as f:
            self._dataset = f.create_dataset('chain', (n_steps, self._n_chains, self._n_dim), chunks=True)

            for _ in tqdm(range(n_steps)):
                self.propose_step()

        self.save_mcmc_chain_to_pdf(output_file_name, "traces.pdf")
