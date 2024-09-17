'''
Interface class for making plots!
'''
from typing import List
from MaCh3_plot_lib.file_handlers import root_file_loader
import MaCh3_plot_lib.plotters as pt
from matplotlib.backends.backend_pdf import PdfPages
import arviz as az

class plotting_interface:
    '''
    full interface object for making plots
    inputs:
        file_loader : root_file_loader instance
    '''
    def __init__(self, file_loader: root_file_loader):
        '''
        Constructor object
        '''
        self._file_loader =  file_loader
        self._plotter_object_dict = {} # dict of objects from plotting tools

    
    def initialise_new_plotter(self, new_plotter: pt.plotter_base._plotting_base_class , plot_label: str)->None:
        '''
        Adds new plot object to our array
        inputs :
            new_plotter : plotting object
            plot_label : [type=str], how do we want to call this plot?
        '''
        self._plotter_object_dict[plot_label] = new_plotter
    
    def set_credible_intervals(self, credible_intervals: List[float])->None:
        '''
        Sets set of credible intervals across all plots
        inputs :
            credible_intervals : [type=list[int]] sets up a list of credible intervals
        '''

        print(f"Setting credible intervals as {credible_intervals}")

        # bit of defensive programming
        if not isinstance(credible_intervals, list):
            raise ValueError(f"Cannot set credible intervals to {credible_intervals}")
    
        for plotter in list(self._plotter_object_dict.values()):
            if not isinstance(plotter, pt.posterior_base_classes._posterior_plotting_base):
                continue  
            
            # set credible intervals
            plotter.credible_intervals = credible_intervals

    def set_variables_to_plot(self, plot_variables, plot_labels: List[str]=[]):
        '''
        Sets variables we actually want to plot for a subset of plots
        '''
        for plotter in plot_labels:
            self._plotter_object_dict[plotter].plot_params = plot_variables


    def set_is_multimodal(self, param_ids: List[int | str]):
        '''
        Lets posteriors know which parameters are multimodal
        inputs:
            param_ids : list of multi-modal parameter ids/names 
        '''
        print(f"Setting {param_ids} to be multimodal")

        # Loop over our plotters
        for plotter in self._plotter_object_dict.values():
            if not isinstance(plotter, pt._posterior_plotting_base):
                continue   
            
            plotter.set_pars_multimodal(param_ids)


    def set_is_circular(self, param_ids: List[int | str]):
        '''
        Lets posteriors know which parameters are multimodal
        inputs:
            param_ids : list of multi-modal parameter ids/names 
        '''
        print(f"Setting {param_ids} to be circular")

        # Loop over our plotters
        for plotter in self._plotter_object_dict.values():
            if not isinstance(plotter, pt.posterior_base_classes._posterior_plotting_base):
                continue   
            
            plotter.set_pars_circular(param_ids)

    def add_text_to_plots(self, text: str, location: tuple=(0.05, 0.95)):
        for plotter in self._plotter_object_dict.values():
            plotter.add_text_to_figures(text, location)

    def make_plots(self, output_file_name: str, plot_labels: List[str] | str):
        '''
        Outputs all plots from a list of labels to an output PDF
        inputs : 
            output_file_name : [str] -> Output file pdf 
            plot_labels : names of plots in self._plotter_object_dict
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
        Print stats summary to terminal and output as a LaTeX table [text file]
        inputs :
            latex_output_name : [type=str, optional] name of output file
        '''
        summary = az.summary(self._file_loader.ttree_array, kind='stats', hdi_prob=0.9)
        if latex_output_name is None:
            return

        print(summary)

        with open(latex_output_name, "w") as output_file:
            output_file.write(summary.to_latex())
            