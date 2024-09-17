from MaCh3_plot_lib.file_handlers import root_file_loader
import arviz as az
from matplotlib import pyplot as plt
import numpy as np
from MaCh3_plot_lib.plotters.posteriors.posterior_base_classes import _posterior_plotting_base
from matplotlib.figure import Figure

'''
Plotting class for 1D plots. Slightly overcomplicated by various special cases but works a treat!
'''
class posterior_plotter_1D(_posterior_plotting_base):
    def __init__(self, file_loader: root_file_loader)->None:
        '''
        Constructor
        '''
        # Inherit the abstract base class
        super().__init__(file_loader)

    def _generate_plot(self, parameter_name: str) -> Figure:
        '''
        Generates a single posterior plot for a parameter
        
        inputs :
            parameter name : [type=str] Name of parameter
        '''
        if not isinstance(parameter_name, str):
            raise ValueError("Can only pass single parameters to posterior plotting class")

        # Checks if parameter is in our array+gets the index
        param_index = self._get_param_index_from_name(parameter_name)

        fig, axes = plt.subplots(figsize=(30, 30))
        # Set bin number and range
        n_bins = 50
        # Grab our density plot
        # plt.rcParams['image.cmap'] = 'Purples' # To avoid any multi-threaded weirdness
        line_colour_generator = iter(plt.cm.Purples(np.linspace(0.4, 0.8, len(self.credible_intervals)+1)))
        line_colour = next(line_colour_generator)
        hist_kwargs={'density' : True, 'bins' : n_bins, "alpha": 1.0, 'linewidth': None, 'edgecolor': line_colour, 'color': 'white'}
        
        _, bins, patches =  axes.hist(self._file_loader.ttree_array[parameter_name].to_numpy()[0], **hist_kwargs)
        # Make lines for each CI

        cred_rev = self.credible_intervals[::-1]
        for credible in cred_rev:
            line_colour = next(line_colour_generator)

            # We want the bayesian credible interval, for now we set the maximum number of modes for multi-modal parameters to 2
            hdi = az.hdi(self._file_loader.ttree_array, var_names=[parameter_name], hdi_prob=credible,
                        multimodal=self._parameter_multimodal[param_index], max_modes=20, circular=self._circular_params[param_index])
            
            # Might be multimodal so we want all our credible intervals in a 1D array to make plotting easier!
            credible_bounds = hdi[parameter_name].to_numpy()
            
            if isinstance(credible_bounds[0], float):
                credible_bounds = np.array([credible_bounds]) #when we're not multi-modal we need to be a little careful

            # set up credible interval
            plot_label = f"{100*credible}% credible interval "
            for bound in credible_bounds:
                # Set up plotting options
                # Reduce the plotting array to JUST be between our boudnaries
                mask = (bins>=bound[0]) & (bins<=bound[1])
                if bound[0]>bound[1]:
                    # We need a SLIGHTLY different treatment since we loop around for some parameters (delta_cp)
                    mask = (bins>=bound[0]) | (bins<=bound[1])

                for patch_index in np.where(mask)[0]:
                    patches[patch_index-1].set_facecolor(line_colour)
                    patches[patch_index-1].set_edgecolor(None)
                    patches[patch_index-1].set_label(plot_label)
            
        # add legend
        # Set Some labels
        axes.set_xlabel(parameter_name, fontsize=50)
        axes.tick_params(labelsize=40)
        axes.set_title(f"Posterior Density for {parameter_name}", fontsize=60)
        axes.set_ylim(ymin=0)
        axes.set_ylabel("Posterior Density", fontsize=50)

        # Generate Unique set of labels and handles
        plot_handles, plot_labels = axes.get_legend_handles_labels()
        # Sometimes it loses track of the posterior histogram
        unique_labs = [(h, l) for i, (h, l) in enumerate(zip(plot_handles, plot_labels)) if l not in plot_labels[:i]]
        axes.legend(*zip(*unique_labs), loc=(0.6,0.85), fontsize=35, facecolor="white", edgecolor="black", frameon=True)
        #Stop memory issues
        plt.close()
        return fig # Add to global figure list
