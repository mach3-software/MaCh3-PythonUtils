'''
HI : Class to make autocorrelations and traces, puts them all onto a single plot
'''
import arviz as az
from matplotlib import pyplot as plt
from diagnostics.mcmc_plots.plotter_base import _PlottingBaseClass
from matplotlib.figure import Figure

class AutocorrelationTracePlotter(_PlottingBaseClass):
    def _generate_plot(self, parameter_name: str) -> Figure:
        """Generates auto-correlation and trace plots for a single parameter

        :param parameter_name: Name of parameter to plot
        :type parameter_name: str
        :return: Figure containing auto-correlation and trace plots
        :rtype: Figure
        """
        fig, (trace_ax, autocorr_ax) = plt.subplots(nrows=2, sharex=False) 

        # We want the numpy array containing our parameter
        param_array = self._chain_handler.arviz_tree[parameter_name].to_numpy()[0]

        # Okay now we can plot our trace (might as well!)
        trace_ax.plot(param_array, linewidth=0.05, color='purple')

        #next we need to grab our autocorrelations
        # auto_corr =  sm.tsa.acf(param_array, nlags=total_lags)
        auto_corr = az.autocorr(param_array)
        autocorr_ax.plot(auto_corr, color='purple')
        # Now we do some tidying
        trace_ax.set_ylabel(f"{parameter_name} variation", fontsize=11)
        trace_ax.set_xlabel("Step", fontsize=11)
        trace_ax.set_title(f"Trace for {parameter_name}", fontsize=15)
        trace_ax.tick_params(labelsize=10)
        # autocorr_ax.set_ylabel("Autocorrelation Function")
        autocorr_ax.set_xlabel("Lag", fontsize=12)
        autocorr_ax.set_ylabel("Autocorrelation", fontsize=12)
        autocorr_ax.set_title(f"Autocorrelation for {parameter_name}", fontsize=15, verticalalignment='center_baseline')
        autocorr_ax.tick_params(labelsize=10)
        fig.suptitle(f"{parameter_name} diagnostics")#, fontsize=40)
        # fig.subplots_adjust(hspace=0.01)
        fig.subplots_adjust(wspace=0.0, hspace=0.3)

        plt.close()
        return fig