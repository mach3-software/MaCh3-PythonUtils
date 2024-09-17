'''
HI : Several simple diagnostics
'''

from typing import List
from MaCh3_plot_lib.file_handlers import root_file_loader
import arviz as az
from matplotlib import pyplot as plt
from MaCh3_plot_lib.plotters.plotter_base import _plotting_base_class
from matplotlib.figure import Figure

class effective_sample_size_plotter(_plotting_base_class):
    '''
    Caclulates effective sample size : https://arxiv.org/pdf/1903.08008.pdf
    '''
    def __init__(self, file_loader: root_file_loader)->None:
        # Constructor
        super().__init__(file_loader)

    def _generate_plot(self, parameter_name: str) -> Figure:
        '''
        Makes effective sample size
        inputs : 
            parameter_name : [type=str] Single parameter name
        outputs : 
            Figure
        '''
        fig, axes = plt.subplots()

        az.plot_ess(self._file_loader.ttree_array, var_names=parameter_name,
                     ax=axes, textsize=30, color='purple', drawstyle="steps-mid", linestyle="-")
        plt.close()
        return fig
    
class markov_chain_standard_error(_plotting_base_class):
    '''
    Calculates Markov Chain Standard Error : https://arxiv.org/pdf/1903.08008.pdf
    '''
    def __init__(self, file_loader: root_file_loader)->None:
        # Constructor
        super().__init__(file_loader)

    def _generate_plot(self, parameter_name: str) -> Figure:
        '''
        Makes MCSE
        inputs : 
            parameter_name : [type=str] Single parameter name
        outputs : 
            Figure
        '''
        fig, axes = plt.subplots()
        az.plot_mcse(self._file_loader.ttree_array, var_names=parameter_name, ax=axes, textsize=10, color='purple')
        plt.close()
        return fig


class violin_plotter(_plotting_base_class):
    # Class to generate Violin Plots
    def __init__(self, file_loader: root_file_loader)->None:
        # Constructor
        super().__init__(file_loader)

    def _generate_plot(self, parameter_name: str | List[str]) -> Figure:
        '''
        Generates a plot for a single parameter
        '''
        # total number of axes we need
        fig, axes = plt.subplots()
        az.plot_violin(self._file_loader.ttree_array, 
                       var_names=parameter_name, ax=axes, textsize=10,
                       shade_kwargs={'color':'purple'})
        plt.close()
        return fig