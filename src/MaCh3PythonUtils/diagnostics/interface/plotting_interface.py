'''
Interface class for making plots!
'''
from typing import List
from MaCh3PythonUtils.file_handling.chain_handler import ChainHandler
from MaCh3PythonUtils.diagnostics.mcmc_plots.plotter_base import _PlottingBaseClass
from MaCh3PythonUtils.diagnostics.mcmc_plots.posteriors.posterior_base_classes import _PosteriorPlottingBase
from matplotlib.backends.backend_pdf import PdfPages
import arviz as az

class PlottingInterface:
    def __init__(self, chain_handler: ChainHandler)->None:
        """Interface for handling plotting objects

        :param chain_handler: ChainHandler instance
        :type chain_handler: ChainHandler
        """        
        self._chain_handler =  chain_handler
        self._plotter_object_dict = {} # dict of objects from plotting tools

        self._chain_handler.make_arviz_tree()
    
    def initialise_new_plotter(self, new_plotter: _PlottingBaseClass , plot_label: str)->None:
        """Initialises a new plotter object

        :param new_plotter: Instance of new plotting class
        :type new_plotter: pt.plotter_base._PlottingBaseClass
        :param plot_label: Label for calling this instance
        :type plot_label: str
        """
        self._plotter_object_dict[plot_label] = new_plotter
    
    def set_credible_intervals(self, credible_intervals: List[float])->None:
        """Sets credible intervals for for posterior plots

        :param credible_intervals: List of credivle intervals
        :type credible_intervals: List[float]
        :raises ValueError: Checks if crredible intervals are a valid type
        """

        print(f"Setting credible intervals as {credible_intervals}")

        # bit of defensive programming
        if not isinstance(credible_intervals, list):
            raise ValueError(f"Cannot set credible intervals to {credible_intervals}")
    
        for plotter in list(self._plotter_object_dict.values()):
            if not isinstance(plotter, pt.posterior_base_classes._PosteriorPlottingBase):
                continue  
            
            # set credible intervals
            plotter.credible_intervals = credible_intervals

    def set_variables_to_plot(self, plot_variables: List[str], plot_labels: List[str]=[])->None:
        """Set variables to plot

        :param plot_variables: Variables to plot
        :type plot_variables: List[str]
        :param plot_labels: Plots we want to only plot these variables for, defaults to []
        :type plot_labels: List[str], optional
        """
        for plotter in plot_labels:
            self._plotter_object_dict[plotter].plot_params = plot_variables


    def set_is_multimodal(self, param_ids: List[int | str])->None:
        '''
        [Summary]
        Lets posteriors know which parameters are multimodal
        :param param_ids: list of multi-modal parameter ids/names 
        :type param_ids: List[str]

        '''
        print(f"Setting {param_ids} to be multimodal")

        # Loop over our plotters
        for plotter in self._plotter_object_dict.values():
            if not isinstance(plotter, pt._posterior_plotting_base):
                continue   
            
            plotter.set_pars_multimodal(param_ids)


    def set_is_circular(self, param_ids: List[int | str])->None:
        '''
        [Summary]
        Lets posteriors know which parameters are multimodal
        
        :param param_ids: list of multi-modal parameter ids/names 
        :type param_ids: List[str]
        '''
        print(f"Setting {param_ids} to be circular")

        # Loop over our plotters
        for plotter in self._plotter_object_dict.values():
            if not isinstance(plotter, pt.posterior_base_classes._PosteriorPlottingBase):
                continue   
            
            plotter.set_pars_circular(param_ids)

    def add_text_to_plots(self, text: str, location: tuple=(0.05, 0.95)):
        '''
        [Summary]
        Adds text to some plots
        :param text: Text to add to plot
        :param location: Location of text on plot
        '''
        for plotter in self._plotter_object_dict.values():
            plotter.add_text_to_figures(text, location)

    def make_plots(self, output_file_name: str, plot_labels: List[str] | str):
        '''
        [Summary]
        Outputs all plots from a list of labels to an output PDF
        :param output_file_name: Output file pdf name
        :type output_file_name: str:
        :param plot_labels: names of plots in self._plotter_object_dict
        :type plot_labels: List[str]
        '''
        # Cast our labels to hist
        if not isinstance(plot_labels, list):
            plot_labels = list(plot_labels)

        with PdfPages(output_file_name) as pdf_file:
            for plotter_id in plot_labels:
                try:
                    plotter_obj = self._plotter_object_dict.get(plotter_id)
                except KeyError:
                    print(f"Warning:Key not found {plotter_id}, skipping")
                    continue
                
                print(f"Generating Plots for {plotter_id}")
                plotter_obj.generate_all_plots()
                print(f"Printing to {output_file_name}")
                plotter_obj.write_to_pdf(existing_pdf_fig=pdf_file)

    def print_summary(self, latex_output_name:str=None):
        '''
        [Summary]
        Print stats summary to terminal and output as a LaTeX table [text file]
        :param latex_output_name: name of output file, if empty or None doesn't print to file
        :type latex_output_name: str, optional
        '''
        summary = az.summary(self._chain_handler.arviz_tree, kind='stats', hdi_prob=0.9)
        if latex_output_name is None:
            return

        print(summary)

        with open(latex_output_name, "w") as output_file:
            output_file.write(summary.to_latex())
            