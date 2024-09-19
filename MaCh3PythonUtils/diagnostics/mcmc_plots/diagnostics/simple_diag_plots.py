'''
HI : Several simple diagnostics
'''

from typing import List
from file_handling.chain_handler import ChainHandler
import arviz as az
from matplotlib import pyplot as plt
from diagnostics.mcmc_plots.plotter_base import _PlottingBaseClass
from matplotlib.figure import Figure

class EffectiveSampleSizePlotter(_PlottingBaseClass):
    def _generate_plot(self, parameter_name: str) -> Figure:
        """Generates ESS plot for single parameter

        :param parameter_name: Name of parameter to plot
        :type parameter_name: str
        :return: ESS plot for parameter_name
        :rtype: Figure
        """
        fig, axes = plt.subplots()

        az.plot_ess(self._chain_handler.arviz_tree, var_names=parameter_name,
                     ax=axes, textsize=30, color='purple', drawstyle="steps-mid", linestyle="-")
        plt.close()
        return fig
    
class MarkovChainStandardError(_PlottingBaseClass):
    def _generate_plot(self, parameter_name: str) -> Figure:
        """Generates MCSE plot for single parameter

        :param parameter_name: Name of parameter to plot
        :type parameter_name: str
        :return: ESS plot for parameter_name
        :rtype: Figure
        """
        fig, axes = plt.subplots()
        az.plot_mcse(self._chain_handler.arviz_tree, var_names=parameter_name, ax=axes, textsize=10, color='purple')
        plt.close()
        return fig


class ViolinPlotter(_PlottingBaseClass):
    def _generate_plot(self, parameter_name: str | List[str]) -> Figure:
        """Generates Violin plot for single parameter

        :param parameter_name: Name of parameter to plot
        :type parameter_name: str
        :return: ESS plot for parameter_name
        :rtype: Figure
        """
        # total number of axes we need
        fig, axes = plt.subplots()
        az.plot_violin(self._chain_handler.arviz_tree, 
                       var_names=parameter_name, ax=axes, textsize=10,
                       shade_kwargs={'color':'purple'})
        plt.close()
        return fig