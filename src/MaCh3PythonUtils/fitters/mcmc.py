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
    def __init__(self, interface: FileMLInterface, circular_params: List[str]=[], update_step: int=10, start_matrix_throw: int=0):
        print("MCMC let's go!")
        
        self._interface = interface        
        self._n_dim = interface.chain.ndim - 1
        self._n_chains = 1
        
        # HACK, we fit to scaled rather than anything else because I AM LAZY (and slight efficiency saving since we invert this at the end)
        self._upper_bounds = interface.chain.upper_bounds[:-1]        
        self._lower_bounds = interface.chain.lower_bounds[:-1]
        self._sampler = None

        # Get initial state etc.
        self._chain_state: NDArray = self._interface.training_data.iloc[[1]].to_numpy()[0]
        
        
        self._circular_indices = [self._interface.chain.plot_branches.index(par) for par in circular_params]
        self._matrix_scale = 2.4**2/float(self._n_dim)
        self._start_matrix_throw = start_matrix_throw
        self._update_step = update_step
        
        self._matrix_handler = CovarianceUpdater(self._n_dim, update_step)

        self._current_loglikelihood = self._calc_likelihood(self._chain_state)
        self._current_step = 0

    def _wrap_circular(self, state):
        return (state + np.pi) % (2 * np.pi) - np.pi

    def _calc_likelihood(self, state: NDArray):
        # state[self._circular_indices] = self._wrap_circular(state[self._circular_indices])
        
        # if np.any(state>self._upper_bounds) or np.any(state<self._lower_bounds):
        #     return -np.inf
        
        return -1*self._interface.model_predict_single_sample(state)

    def propose_step(self):
        proposed_state = self._matrix_handler.sample(self._chain_state)
        
        
        proposed_loglikelihood = self._calc_likelihood(proposed_state)
        
        if proposed_loglikelihood > -1*float(np.inf):
            u = np.random.uniform(0, 1)
            
            if proposed_loglikelihood>self._current_loglikelihood:
                acc_prob = 1
            else:
                acc_prob = np.exp(proposed_loglikelihood-self._current_loglikelihood)
            
            if min(1, acc_prob)>u:
                self._chain_state = proposed_state
                self._current_loglikelihood = proposed_loglikelihood
        
        self._matrix_handler.update(self._chain_state[0])
        
        
        self._dataset[self._current_step]=self._chain_state
        self._current_step += 1
  

    def save_mcmc_chain_to_pdf(self, filename: str, output_pdf: str):
        # Open the HDF5 file
        with h5py.File(filename, 'r') as f:
            # Load the chain dataset
            chain = f['chain'][:]

        _, n_params = chain.shape

        # Create a PdfPages object to save plots
        with PdfPages(output_pdf) as pdf:
            for i in tqdm(range(n_params)):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot the chain for the i-th parameter
                ax.plot(chain[:, i], lw=0.5)
                ax.set_ylabel(self._interface.chain.plot_branches[i])
                ax.set_title(f"Parameter {self._interface.chain.plot_branches[i]} MCMC Chain")
                ax.set_xlabel('Step')
                
                # Save the current figure to the PDF
                pdf.savefig(fig)
                plt.close(fig)  # Close the figure to save memory

        print(f"MCMC chain plots saved to {output_pdf}")
          
        
    def __call__(self, n_steps, output_file_name: str):
        print(f"Running MCMC for {n_steps} steps")
        
        # Open the HDF5 file and create the dataset with the correct shape upfront
        with h5py.File(output_file_name, 'w') as f:
            self._dataset = f.create_dataset('chain', (n_steps, self._n_dim), chunks=True)
            
            for _ in tqdm(range(n_steps)):
                self.propose_step()
        
        self.save_mcmc_chain_to_pdf(output_file_name, "traces.pdf")