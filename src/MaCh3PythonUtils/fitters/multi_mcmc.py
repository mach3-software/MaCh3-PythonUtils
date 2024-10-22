import emcee
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
from typing import List
from MaCh3PythonUtils.machine_learning.file_ml_interface import FileMLInterface
from MaCh3PythonUtils.fitters.adaption_handler import CovarianceUpdater

class MCMC:
    def __init__(self, interface: FileMLInterface, n_chains: int = 1024, circular_params: List[str] = [], update_step: int = 10, start_matrix_throw: int = 0):
        print("MCMC let's go!")

        self._interface = interface        
        self._n_dim = interface.chain.ndim - 1
        self._n_chains = n_chains
        
        # HACK, we fit to scaled rather than anything else because I AM LAZY (and slight efficiency saving since we invert this at the end)
        self._upper_bounds = interface.chain.upper_bounds[:-1]        
        self._lower_bounds = interface.chain.lower_bounds[:-1]
        self._sampler = None

        # Get initial states for all chains
        initial_state = self._interface.training_data.iloc[[1]].to_numpy()[0]
        self._chain_states: NDArray = np.full((n_chains, self._n_dim),initial_state)  # Create a state for each chain

        self._circular_indices = [self._interface.chain.plot_branches.index(par) for par in circular_params]
        self._matrix_scale = 2.4**2 / float(self._n_dim)
        self._start_matrix_throw = start_matrix_throw
        self._update_step = update_step

        # Use the first chain to update the covariance matrix
        self._matrix_handler = CovarianceUpdater(self._n_dim, update_step)
        
        # Calculate likelihoods for all chains
        self._current_loglikelihoods = self._calc_likelihood(self._chain_states)
        self._current_step = 0

    def _wrap_circular(self, state):
        return (state + np.pi) % (2 * np.pi) - np.pi

    def _calc_likelihood(self, states: NDArray):
        # states[:, self._circular_indices] = self._wrap_circular(states[:, self._circular_indices])
        # Apply boundary checks
        # Apply model prediction to each chain
        return -1 * self._interface.model_predict(states)

    def propose_step(self):
        # Propose steps for all chains
        proposed_states = self._matrix_handler.sample(self._n_chains)+self._chain_states

        # Calculate likelihoods for all proposed states (batch processing)
        proposed_loglikelihoods = self._calc_likelihood(proposed_states)

        # Metropolis-Hastings acceptance step (vectorized)
        valid_loglikelihoods = proposed_loglikelihoods > -1 * float(np.inf)
        
        # Compute acceptance probabilities
        log_diff = proposed_loglikelihoods - self._current_loglikelihoods
        acc_probs = np.minimum(1, np.exp(np.clip(log_diff, -100, 0)))  # clip to avoid overflow

        # Generate uniform random numbers for comparison
        u = np.random.uniform(0, 1, size=self._n_chains)
        
        # Update chains where proposed step is accepted
        accept = (acc_probs > u) & valid_loglikelihoods
        self._chain_states[accept] = proposed_states[accept]
        self._current_loglikelihoods[accept] = proposed_loglikelihoods[accept]

        # Update covariance matrix only based on the first chain
        self._matrix_handler.update(self._chain_states[0])

        # Save the current state of all chains
        self._dataset[self._current_step, :] = self._chain_states
        self._current_step += 1

    def save_mcmc_chain_to_pdf(self, filename: str, output_pdf: str):
        # Open the HDF5 file
        with h5py.File(filename, 'r') as f:
            # Load the chain dataset
            chain = f['chain'][:]

        _, n_params = chain.shape[1:]

        # Create a PdfPages object to save plots
        with PdfPages(output_pdf) as pdf:
            for i in tqdm(range(n_params)):
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot the chain for the i-th parameter
                for j in range(self._n_chains):
                    ax.plot(chain[:, j, i], lw=0.5, label=f'Chain {j+1}')
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

        # Open the HDF5 file and create the dataset with the correct shape upfront
        with h5py.File(output_file_name, 'w') as f:
            self._dataset = f.create_dataset('chain', (n_steps, self._n_chains, self._n_dim), chunks=True)

            for _ in tqdm(range(n_steps)):
                self.propose_step()

        self.save_mcmc_chain_to_pdf(output_file_name, "traces.pdf")
