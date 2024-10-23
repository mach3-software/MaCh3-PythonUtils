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
import psutil  # To get system memory information
from MaCh3PythonUtils.machine_learning.tf_interface import TfInterface
from MaCh3PythonUtils.fitters.adaption_handler_gpu import CovarianceUpdaterGPU

class MCMCMultGPU:
    def __init__(self, interface: TfInterface, n_chains: int = 1024, circular_params: List[str] = [], update_step: int = 10):
        print("MCMC let's go!")

        self._interface = interface        
        self._n_dim = interface.chain.ndim - 1
        self._n_chains = n_chains

        # Initial states for all chains
        initial_state = tf.convert_to_tensor(np.ones(self._n_dim), dtype=tf.float32)
        self._chain_states = tf.Variable(tf.tile(tf.expand_dims(initial_state, axis=0), [n_chains, 1]), dtype=tf.float32)

        # boundaries
        self._upper_bounds = tf.convert_to_tensor(self._interface.scale_data(self._interface.chain.upper_bounds[:-1].reshape(1,-1)), dtype=tf.float32)
        self._lower_bounds = tf.convert_to_tensor(self._interface.scale_data(self._interface.chain.lower_bounds[:-1].reshape(1,-1)), dtype=tf.float32)


        self._circular_indices = [self._interface.chain.plot_branches.index(par) for par in circular_params]
        # CovarianceUpdater will be updated based on the first chain
        self._matrix_handler = CovarianceUpdaterGPU(self._n_dim, update_step)

        # Calculate likelihoods for all chains (with GPU processing)
        self._current_loglikelihoods = tf.Variable(self._calc_likelihood(self._chain_states))
        self._current_step = 0

        # Automate batch size determination based on memory availability for this process
        self._batch_size_steps = self._estimate_batch_size()

        # Use a FIFOQueue to cache steps asynchronously
        self._queue = tf.queue.FIFOQueue(
            capacity=self._batch_size_steps,
            dtypes=[tf.float32],
            shapes=[(self._n_chains, self._n_dim)]
        )

    def _estimate_batch_size(self):
        """Estimate batch size based on memory available to this process."""
        step_size_in_bytes = self._n_chains * self._n_dim * tf.float32.size
        
        # Get memory info for the current process
        process = psutil.Process()
        available_memory = process.memory_info().rss  # Get memory in use by the process
        memory_fraction = 0.08  # Use 8% of available memory for caching steps
        usable_memory = available_memory * memory_fraction

        estimated_batch_size = int(usable_memory // step_size_in_bytes)
        print(f"Automatically determined batch size for the process: {estimated_batch_size}")

        return estimated_batch_size

    def _wrap_circular(self, state):
        """Apply circular boundary conditions."""
        return (state + np.pi) % (2 * np.pi) - np.pi

    @tf.function
    def _calc_likelihood(self, states: tf.Tensor):
        return -1 * self._interface.model_predict_no_scale(states)

    @tf.function
    def propose_step_gpu(self):
        # Propose new states for all chains
        proposed_states = self._matrix_handler.sample(self._n_chains) + self._chain_states

        # Apply boundary conditions
        # proposed_states = tf.where(proposed_states < self._lower_bounds, self._chain_states, proposed_states)
        # proposed_states = tf.where(proposed_states > self._upper_bounds, self._chain_states, proposed_states)

        # Calculate log-likelihoods for proposed states
        proposed_loglikelihoods = self._calc_likelihood(proposed_states)

        # Metropolis-Hastings acceptance step
        log_diff = proposed_loglikelihoods - self._current_loglikelihoods
        acc_probs = tf.minimum(1.0, tf.exp(tf.clip_by_value(log_diff, -100, 0)))

        # Generate uniform random values for comparison
        u = tf.random.uniform(shape=(self._n_chains, 1), dtype=tf.float32)

        # Determine which states to accept (True or False for each chain)
        accept = acc_probs > u

        # Use `tf.where` to conditionally update the chain states based on acceptance
        self._chain_states.assign(tf.where(accept, proposed_states, self._chain_states))
        self._current_loglikelihoods.assign(tf.where(accept, proposed_loglikelihoods, self._current_loglikelihoods))

        # Update covariance based on first chain
        self._matrix_handler.update(self._chain_states[0])

    def propose_step(self):
        # Perform GPU step proposal
        self.propose_step_gpu()

        # Cache the current chain states in the FIFO queue
        self._queue.enqueue(self._chain_states)

        # Increment the current step counter
        self._current_step += 1

        # Automate flushing when batch size is reached
        if self._current_step % self._batch_size_steps == 0:
            self._flush_async()

    def _flush_async(self, final_flush=False):
        """
        Dequeue cached steps in a single batch and write to the HDF5 file.
        
        If `final_flush` is True, flush the remaining items even if they don't
        fill a full batch.
        """
        if final_flush:
            # Get the current queue size
            queue_size = self._queue.size().numpy()

            if queue_size > 0:
                # Dequeue remaining steps if queue is not empty
                steps_to_write = self._queue.dequeue_many(queue_size)
                start_idx = self._current_step - queue_size
                end_idx = self._current_step

                # Write the remaining steps to the HDF5 file
                self._dataset[start_idx:end_idx, :] = steps_to_write
        else:
            # Normal flush: dequeue exactly `batch_size_steps`
            steps_to_write = self._queue.dequeue_many(self._batch_size_steps)
            end_idx = self._current_step

            self._dataset[:end_idx, :] = steps_to_write

    def save_mcmc_chain_to_pdf(self, filename: str, output_pdf: str):
        # Open the HDF5 file and read the chain
        with h5py.File(filename, 'r') as f:
            chain = f['chain'][:]

        _, n_params = chain.shape[1:]
        
        # Create a PdfPages object to save plots
        with PdfPages(output_pdf) as pdf:
            for i in tqdm(range(n_params)):
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot the chain for the i-th parameter
                # unscaled_data = self._interface.invert_scaling(chain[:, 0, i])
                
                ax.plot(chain[:, 0, i], lw=0.5, label=f'Chain {i}')
                ax.set_ylabel(self._interface.chain.plot_branches[i])
                ax.set_title(f"Parameter {self._interface.chain.plot_branches[i]} MCMC Chain")
                ax.set_xlabel('Step')

                # Save the current figure to the PDF
                pdf.savefig(fig)
                plt.close(fig)  # Close the figure to save memory

        print(f"MCMC chain plots saved to {output_pdf}")

    def __call__(self, n_steps, output_file_name: str):
        print(f"Running MCMC for {n_steps} steps with {self._n_chains} chains")

        # Open the HDF5 file in append mode
        with h5py.File(output_file_name, 'w') as f:
            # Create or resize the dataset
            self._dataset = f.create_dataset('chain', (n_steps, self._n_chains, self._n_dim), chunks=True)

            for _ in tqdm(range(n_steps)):
                self.propose_step()

            # Ensure remaining steps are flushed to disk
            self._flush_async(final_flush=True)

            # Save the MCMC chain to PDF
            self.save_mcmc_chain_to_pdf(output_file_name, "traces.pdf")
