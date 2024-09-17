'''
HI : Several simple diagnostics
'''

from typing import List
from file_handling.chain_handler import ChainHandler
import arviz as az
from matplotlib import pyplot as plt
from diagnostics.plotters.plotter_base import _PlottingBaseClass
from matplotlib.figure import Figure

class EffectiveSampleSizePlotter(_PlottingBaseClass):
    '''
    Caclulates effective sample size : https://arxiv.org/pdf/1903.08008.pdf
    '''
    def __init__(self, file_loader: ChainHandler)->None:
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

        az.plot_ess(self._file_loader.arviz_tree, var_names=parameter_name,
                     ax=axes, textsize=30, color='purple', drawstyle="steps-mid", linestyle="-")
        plt.close()
        return fig
    
class MarkovChainStandardError(_PlottingBaseClass):
    '''
    Calculates Markov Chain Standard Error : https://arxiv.org/pdf/1903.08008.pdf
    '''
    def __init__(self, file_loader: ChainHandler)->None:
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
        az.plot_mcse(self._file_loader.arviz_tree, var_names=parameter_name, ax=axes, textsize=10, color='purple')
        plt.close()
        return fig


class ViolinPlotter(_PlottingBaseClass):
    # Class to generate Violin Plots
    def __init__(self, file_loader: ChainHandler)->None:
        # Constructor
        super().__init__(file_loader)

    def _generate_plot(self, parameter_name: str | List[str]) -> Figure:
        '''
        Generates a plot for a single parameter
        '''
        # total number of axes we need
        fig, axes = plt.subplots()
        az.plot_violin(self._file_loader.arviz_tree, 
                       var_names=parameter_name, ax=axes, textsize=10,
                       shade_kwargs={'color':'purple'})
        plt.close()
        return fig