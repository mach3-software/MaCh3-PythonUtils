'''
HI : Makes 2D posterior plots. Currently no way of putting a legend on the plot (thanks arviz...)
'''
from config_reader import ChainHandler
import arviz as az
from typing import List, Any
from matplotlib import pyplot as plt
from itertools import combinations
from diagnostics.plotters.posteriors.posterior_base_classes import _PosteriorPlottingBase
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt

class PosteriorPlotter2D(_PosteriorPlottingBase):
    # For making 2D possteriors
    def __init__(self, file_loader: ChainHandler)->None:
        '''
        Constructor
        '''
        # Inherit the abstract base class
        super().__init__(file_loader) 
    
    def _generate_plot(self, parameter_names: List[str]) -> npt.NDArray[Any]:
        '''
        Generates a 2D posterior plot
        inputs :
            -> Parameter Names [type=List[str]] list of parameters, will plot all combinations of pairs

        returns :
            -> figure
        '''
        # Let's get pairs of names
        name_pairs_list = list(combinations(parameter_names, 2))
        fig_list = np.empty(len(name_pairs_list), dtype=Figure)
        # Now we loop over our pairs of names
        for i, (par_1, par_2) in enumerate(name_pairs_list):
            
            fig, axes = plt.subplots(figsize=(30, 30))

            par_1_numpy_arr = self._file_loader.arviz_tree[par_1].to_numpy()[0]
            par_2_numpy_arr = self._file_loader.arviz_tree[par_2].to_numpy()[0]

            ciruclar = self._circular_params[self._get_param_index_from_name(par_1)] | self._circular_params[self._get_param_index_from_name(par_2)]

            az.plot_kde(par_1_numpy_arr, par_2_numpy_arr,
                    hdi_probs= self._credible_intervals,
                    contourf_kwargs={"cmap": "Purples"},
                    ax=axes, is_circular=ciruclar, legend=True)

            axes.set_xlabel(par_1, fontsize=50)
            axes.set_ylabel(par_2, fontsize=50)
            axes.tick_params(labelsize=40)
            

            cred_as_str=",".join(f"{100*i}%" for i in self._credible_intervals)
            axes.set_title(f"{par_1} vs {par_2} : [{cred_as_str}] credible intervals", fontsize="60")
            plt.close() # CLose the canvas
            fig_list[i] = fig
        return fig_list



class TrianglePlotter(_PosteriorPlottingBase):
    # Makes triangle plots
    def __init__(self, file_loader: ChainHandler)->None:
        '''
        Constructor
        '''
        # Inherit the abstract base class
        super().__init__(file_loader)

    def _generate_plot(self, parameter_names: List[str]) -> Figure:
        '''
        Generates a single triangle plot and adds it to the figure list
        inputs :
            -> parameter_names : [List[str]] List of parameter names to put in the triangle
        returns :
            -> Figure
        '''
        if not isinstance(parameter_names, list):
            raise ValueError("Parameter names must be list when plotting triangle plots")
        
        # Check we are using a valid parameter set
        for param in parameter_names:
            self._parameter_not_found_error(param) # Check if parameters exist
        
        fig, axes = plt.subplots(nrows=len(parameter_names), ncols=len(parameter_names), figsize=(30, 30))

        az.plot_pair(self._file_loader.arviz_tree, var_names=parameter_names,         
            marginals=True, ax=axes, colorbar=True, figsize=(30,30),
            kind='kde',
            textsize=30,
            kde_kwargs={
                "hdi_probs": self._credible_intervals,  # Plot HDI contours
                "contourf_kwargs": {"cmap": "Purples"},
                'legend':True
            },
            marginal_kwargs={
                # 'fill_kwargs': {'alpha': 0.0},
                'plot_kwargs': {"linewidth": 4.5, "color": "purple"},
                # "quantiles": credible_intervals,
                "rotated": False
            },
            point_estimate='mean',
        )

        cred_as_str=",".join(f"{100*i}%" for i in self._credible_intervals)
        fig.suptitle(f"Triangle Plot for : {cred_as_str} credible intervals", fontsize="60")

        # axes[-1, -1].legend(axes, [f"{i}% Credible Interval" for i in self._credible_intervals], frameon=True, loc='right')

        fig.subplots_adjust(wspace=0.01, hspace=0.01)

        plt.close()
        return fig