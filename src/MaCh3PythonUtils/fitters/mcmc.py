import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import emcee
import corner
import seaborn as sns
from typing import List


from MaCh3PythonUtils.machine_learning.file_ml_interface import FileMLInterface


class CyclicMove(emcee.moves.Move):
    def __init__(self, param_index, low=-np.pi, high=np.pi, sigma=0.1):
        self.param_index = param_index
        self.low = low
        self.high = high
        self.sigma = sigma

    def get_proposal(self, coords, random):
        new_coords = coords.copy()
        step = random.normal(0, self.sigma, size=coords.shape)
        new_coords[:, self.param_index] += step[:, self.param_index]
        # Apply cyclical wrapping
        new_coords[:, self.param_index] = (new_coords[:, self.param_index] - self.low) % (self.high - self.low) + self.low
        return new_coords, np.zeros(coords.shape[0])


class MCMC:
    def __init__(self, interface: FileMLInterface, boundary_expansion=0.1, circular_params: List[str]=[]):
        print("MCMC let's go!")
        
        self._interface: FileMLInterface = interface        
        self._n_dim = interface.chain.ndim - 1
        
        # HACK, slightly increase boundary so we can explore slightly more of the space
        self._upper_bounds = interface.chain.upper_bounds[:-1] + np.abs(interface.chain.upper_bounds[:-1]*boundary_expansion)
        self._lower_bounds = interface.chain.lower_bounds[:-1] - np.abs(interface.chain.lower_bounds[:-1]*boundary_expansion)        
        self._sampler = None
        
        circular_indices = [self._interface.chain.plot_branches.index(par) for par in circular_params]        
    
    def calc_loglikelihood(self, input_vals: NDArray):
        
        # Make life easier
        if np.any(input_vals<self._lower_bounds) or np.any(input_vals>self._upper_bounds):
            return -1*np.inf
        
        # Reverse it
        
        # Specifically for delm2_32
        return -1*self._interface.evaluate_model(input_vals)
    
        
    def get_plots(self):
        if self._sampler is None:
            return
        
        flat_samples = self._sampler.get_chain(discard=1000, flat=True, thin=10)

        print("Making posterior plots!")
        with PdfPages("posteriors.pdf") as pdf:
            for param in tqdm(range(self._n_dim)):
                plt.figure(figsize=(8, 6))
                sns.kdeplot(flat_samples[:, param], bw_adjust=0.5, fill=True, alpha=0.5)
                plt.title(f"Posterior for {self._interface.chain.plot_branches[param]}")
                plt.xlabel(f"{self._interface.chain.plot_branches[param]}")
                plt.ylabel("Density")
                pdf.savefig()
                plt.close()

    def __call__(self, n_steps: int, n_walkers: int):
        # Setup backend
        filename = "emcee_chain.h5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(n_walkers, self._n_dim)

        # Make sampler
        self._sampler = emcee.EnsembleSampler(n_walkers, self._n_dim, self.calc_loglikelihood, backend=backend)
        # Initialise it
        init_state = self._interface.training_data.iloc[[1]].to_numpy()[0]
        
        # Set walkers in random positions
        
        p0 = [init_state+np.random.uniform(self._lower_bounds, self._upper_bounds) for _ in range(n_walkers)]        
        # Run it
        self._sampler.run_mcmc(p0, n_steps, progress=True)
        # Grab plots
        self.get_plots()
