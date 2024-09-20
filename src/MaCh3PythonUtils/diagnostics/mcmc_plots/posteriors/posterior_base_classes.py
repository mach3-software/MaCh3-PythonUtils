'''
HI : Contains a set of base classes with methods common to posterior plotting code
Separated into 1D and 2D-like objects to make life easier
'''
from MaCh3PythonUtils.file_handling.chain_handler import ChainHandler
from MaCh3PythonUtils.diagnostics.mcmc_plots.plotter_base import _PlottingBaseClass
from typing import List
import numpy as np
from tqdm.auto import tqdm
import numpy.typing as npt


# Base class for all posterior plotters
class _PosteriorPlottingBase(_PlottingBaseClass):
    # Small extension of _plotting_base class for posterior specific stuff
    def __init__(self, file_loader: ChainHandler)->None:
        """Base class for all posterior plotters

        :param file_loader: ChainHandler instance
        :type file_loader: ChainHandler
        """        
        # Setup additional features for posteriors
        super().__init__(file_loader)
        self._credible_intervals = np.array([0.6, 0.9, 0.95])
        self._parameter_multimodal = np.zeros(len(self._file_loader.arviz_tree)).astype(bool)
    
    def __str__(self) -> str:
        return "posterior_plotting_base_class"

    @property
    def credible_intervals(self)->List[float]:
        """Get Credible Intervals

        :return: List of credible intervals
        :rtype: List[float]
        """        
        return self._credible_intervals
    
    @credible_intervals.setter
    def credible_intervals(self, new_creds: List[float])->None:
        """Set new credible intervals

        :param new_creds: Sets list of credible intervals
        :type new_creds: List[float]
        """        
        # Sets credible intervals from list
        self._credible_intervals = np.array(new_creds) # Make sure it's 1D
        self._credible_intervals.sort() # Flatten it

    def set_pars_multimodal(self, par_id_list: List[str] | List[int], is_multimodal: bool=True)->None:
        """Let arviz know which parameters are multi-modal

        :param par_id_list: List of multimodal parameter names/ids
        :type par_id_list: List[str] | List[int]
        :param is_multimodal: Sets parameters to either be multimodal [True] or uni-modal [False], defaults to True
        :type is_multimodal: bool, optional
        """
        # If it's an int this is easy
        if not isinstance(par_id_list, list):
            par_id_list = list(par_id_list)

        for par_id in par_id_list:
            if isinstance(par_id, int):
                true_index = par_id
            else:
                true_index = self._get_param_index_from_name(par_id)

            self._parameter_multimodal[true_index] = is_multimodal
