'''
HI : Mostly abstract base class, contains common methods for use by most other plotting classes
'''

from file_handling.chain_handler import ChainHandler
import arviz as az
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import List
import mplhep as hep
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy.typing as npt
from typing import Any
from collections.abc import Iterable


# Base class with common methods
class _PlottingBaseClass(ABC):
    '''
    Abstract class with common methods for creating plots
    inputs:
        file_loader : [type=ChainHandler] ChainHandler class instance
    '''
    def __init__(self, file_loader: ChainHandler)->None:
        '''
        Constructor
        '''
        # I'm very lazy but this should speed up parallelisation since we won't have this warning
        plt.rcParams.update({'figure.max_open_warning': 0})        
        
        self._file_loader=file_loader # Make sure we don't use crazy amount of memory
        # Setup plotting styles
        az.style.use(hep.style.ROOT)
        plt.style.use(hep.style.ROOT)

        self._figure_list = np.empty([]) # List of all figures

        # Let's just ignore some warnings :grin:
        self._all_plots_generated = False # Stops us making plots multiple times

        # Setting it like this replicates the old behaviour!
        self._circular_params=np.ones(len(self._file_loader.arviz_tree)).astype(bool)
        warnings.filterwarnings("ignore", category=DeprecationWarning) #Some imports are a little older
        warnings.filterwarnings("ignore", category=UserWarning) #Some imports are a little older

        az.utils.Dask().enable_dask(dask_kwargs={"dask": "parallelized", "output_dtypes": [float]}) # let it be a bit cleverer
        az.utils.Numba().enable_numba()

        # Do we want figure text?
        self._figure_text = None
        self._text_location = (0.05, 0.85)

        # Default option
        self._params_to_plot = list(self._file_loader.arviz_tree.keys())
    
    def __str__(self) -> str:
        return "plotting_base_class"

    @property
    def figure_list(self)->npt.NDArray[Any]:
        return self._figure_list

    @figure_list.setter
    def figure_list(self, new_figure)->None:
        raise NotImplementedError("Cannot set figure list using property!")

    @abstractmethod
    def _generate_plot(self, parameter_name: str | List[str])->None:
        # Abstract method to generate a single plot
        pass

    def generate_all_plots(self)->None:
        # Generates plots for every parameter [can be overwritten]
        if self._all_plots_generated: # No need to make all possible plots again!
            return

        self._figure_list = np.empty(len(self._params_to_plot), dtype=Figure)

        # Parallelised loop
        with ThreadPoolExecutor() as executor:
            # Set of threadpools
            futures = {executor.submit(self._generate_plot, param) : param for param in self._params_to_plot}
            # Begin loop
            for future in tqdm(as_completed(futures), ascii="▖▘▝▗▚▞█", total=len(self._params_to_plot)):
                param_id = list(futures).index(future) # Unique plot ID
                self._figure_list[param_id] = future.result()
                plt.close()

        self._all_plots_generated=True

    # Lets us select a subset/add list of parameters we'd like to plot
    @property
    def plot_params(self)-> List[str] | List[List[str]]:
        '''
        Getter for parameters we want to plot
        '''
        return self._params_to_plot
    
    @plot_params.setter
    def plot_params(self, new_plot_parameter_list: List[str]|List[List[str]]):
        if len(new_plot_parameter_list)==0:
            raise ValueError("Parameter list cannot have length 0")

        # Check our new parameters are in our list of keys
        if isinstance(new_plot_parameter_list[0], str):
            for parameter in new_plot_parameter_list:
                self._parameter_not_found_error(parameter)
            
        
        elif isinstance(new_plot_parameter_list[0][0], str):
            for param_list in new_plot_parameter_list:
                for param in param_list: 
                    self._parameter_not_found_error(param)
            
        else:
            raise ValueError("Plot params must be of type List[str] or List[List[str]]")

        self._params_to_plot = new_plot_parameter_list



    def _parameter_not_found_error(self, parameter_name: str):
        if parameter_name not in list(self._file_loader.arviz_tree.keys()):
            raise ValueError(f"{parameter_name} not in list of parameters!")

    def _get_param_index_from_name(self, parameter_name: str)->int:
        # Gets index of parameter in our arviz array

        self._parameter_not_found_error(parameter_name)
        param_id = list(self._file_loader.arviz_tree.keys()).index(parameter_name)
        return param_id

    def set_pars_circular(self, par_id_list: List[str] | List[int])->None:
        '''
        Let the plotter know parameter set is cyclical
        inputs:
            par_id_list : List[str/int] List of Parameter indices or name
        '''
        # If it's an int this is easy
        if not isinstance(par_id_list, list):
            par_id_list = list(par_id_list)

        for par_id in par_id_list:
            if isinstance(par_id, int):
                true_index = par_id
            else:
                true_index = self._get_param_index_from_name(par_id)

            self._circular_params[true_index] = True
    
    def add_text_to_figures(self, text: str, text_location: tuple=(0.05, 0.95))->None:
        '''
        Add text to all figures
        inputs:
            text : Text to add
            text_location : location of text
        '''
        self._figure_text = text
        self._text_location = text_location

    def write_to_pdf(self, output_pdf_name: str=None, existing_pdf_fig: PdfPages=None)->None:
        '''
        Dump all our plots to PDF file must either set output name or existing_pdf_fig

        inputs : 
            output_pdf_name: [type=string] Output name for NEW pdf
            existing_pdf_fig: [type=PDFPages] 
        returns :
            pdf_fig: [type=PDFPages] pdf file reader [REMEMBER TO CLOSE THE PDF FILE AT THE END!]
        '''
        if len(self._figure_list)==0:
            return

        if output_pdf_name is not None and existing_pdf_fig is None:
            pdf_file = PdfPages(output_pdf_name)
        elif existing_pdf_fig is not None:
            pdf_file = existing_pdf_fig
        else:
            raise ValueError("ERROR:Must set EITHER output_pdf_name OR existing_pdf_fig")

        # For some arrays we might want to make them 1D
        if not isinstance(self._figure_list[0], Figure):
            self._figure_list = [fig for sublist in self._figure_list for fig in sublist]


        for fig in tqdm(self._figure_list, ascii=" ▖▘▝▗▚▞█"):
            # Add text to all plots!
            if self._figure_text is not None:
                if len(fig.get_axes())==1:
                    fig.text(*self._text_location, self._figure_text, transform=fig.get_axes()[0].transAxes, fontsize=60, fontstyle='oblique')

            pdf_file.savefig(fig)
            plt.close()
