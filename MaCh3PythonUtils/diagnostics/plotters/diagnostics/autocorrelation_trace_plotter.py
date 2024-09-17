'''
HI : Class to make autocorrelations and traces, puts them all onto a single plot
'''
from file_handling.chain_handler import ChainHandler
import arviz as az
from matplotlib import pyplot as plt
from diagnostics.plotters.plotter_base import _PlottingBaseClass
from matplotlib.figure import Figure

class AutocorrelationTracePlotter(_PlottingBaseClass):
    def __init__(self, file_loader: ChainHandler)->None:
        # Constructor
        super().__init__(file_loader)

    def _generate_plot(self, parameter_name: str) -> Figure:
        '''
        Makes a combined trace and auto-correlation plot
        inputs : 
        parameter_name : [type=str] Single parameter name
        '''
        # Setup axes
        fig, (trace_ax, autocorr_ax) = plt.subplots(nrows=2, sharex=False) 

        # We want the numpy array containing our parameter
        param_array = self._file_loader.arviz_tree[parameter_name].to_numpy()[0]

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