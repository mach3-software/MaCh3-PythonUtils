'''
Additional Diagnostics that can be used with MCMC but don't rely on plotting

Suboptimality : https://www.jstor.org/stable/25651249?seq=3
'''

from MaCh3_plot_lib.file_handlers import root_file_loader
import numpy as np
from scipy.linalg import sqrtm
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import mplhep as hep
from concurrent.futures import ThreadPoolExecutor, as_completed


class covariance_matrix_utils:
    def __init__(self, file_loader: root_file_loader)->None:
        '''
        For calculating the covariance matrix + suboptimality
        inputs:
            ->file_loader : [type=root_file_loader] file handler object
        '''
        # Let's just ignore some warnings :grin:
        warnings.filterwarnings("ignore", category=DeprecationWarning) #Some imports are a little older
        warnings.filterwarnings("ignore", category=UserWarning) #Some imports are a little older

        file_loader.get_ttree_as_arviz() # Ensure that we have things in the correct format
        self._parameter_names = list(file_loader.ttree_array.keys())
        self._total_parameters = len(self._parameter_names)
        self._ttree_data_frame = file_loader.ttree_array.to_dataframe() # All our handling will be done using thi s object
        # Various useful class properties
        self._suboptimality_array = [] # For filling with suboptimality
        self._suboptimality_evaluation_points = [] #Where did we evalutate this?
        self._full_covariance = self.calculate_covariance_matrix()
        self._sqrt_full_covariance_inv = np.linalg.inv(sqrtm(self._full_covariance))
        plt.style.use(hep.style.ROOT)

    def calculate_covariance_matrix(self, min_step: int=0, max_step: int=-1)->np.ndarray:
        '''
        Calculates covariance matrix for chain between indices min_step and max_step
        inputs:
            -> min_step : [type=int] minimum index to calculate covariance from
            -> max_step : [type=int] maximum index to calculate covariance to
        returns
            -> covariance matrix
        '''
        if(max_step<=0):
            sub_array = self._ttree_data_frame[min_step : ]
        else:
            sub_array = self._ttree_data_frame[min_step : max_step]


        return sub_array.cov().to_numpy()

    def _calculate_matrix_suboptimality(self, step_number: int)->float:
        '''
        Calcualte suboptimality for a given covariance matrix
        inputs:
            -> how many steps in are we calculating this??
        returns:
            -> suboptimality value
        '''
        new_covariance = self.calculate_covariance_matrix(max_step=step_number)

        sqrt_input_cov = sqrtm(new_covariance)

        # Get product of square roots
        matrix_prod = sqrt_input_cov @ self._sqrt_full_covariance_inv
        
        #Get eigen values 
        eigenvalues, _ = np.linalg.eig(matrix_prod)
        
        return self._total_parameters * np.sum(eigenvalues**(-2))/((np.sum(eigenvalues)**(-1))**2)

    def calculate_suboptimality(self, step_skip : int = 1000, min_step=0)->None:
        '''
        Calculates the suboptimalit for every step_skip steps
        inputs :
            -> step_skip : Number of steps to skip
            -> min_step : smallest number of starting steps
        '''
        self._suboptimality_evaluation_points = np.arange(min_step, len(self._ttree_data_frame), step_skip)
        # Make sure we have the last step as well!
        self._suboptimality_evaluation_points = np.append(self._suboptimality_evaluation_points, len(self._ttree_data_frame)-1)
        # make array to fill with suboptimality values
        self._suboptimality_array =  np.zeros(len(self._suboptimality_evaluation_points)) # Reset our suboptimality values
        print(f"Calculating suboptimality for {len(self._suboptimality_evaluation_points)} points")

        # Lets speed this up a bit!
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._calculate_matrix_suboptimality, step): step
                       for step in self._suboptimality_evaluation_points}

            for future in tqdm(as_completed(futures), ascii="▖▘▝▗▚▞█", total=len(self._suboptimality_evaluation_points)):
                i = np.where(self._suboptimality_evaluation_points==futures[future])[0]
                self._suboptimality_array[i]=future.result()
        

    def plot_suboptimality(self, output_file: str):
        print(f"Saving to {output_file}")
        with PdfPages(output_file) as pdf:
            fig, axes = plt.subplots()

            axes.plot(self._suboptimality_evaluation_points, self._suboptimality_array)
            axes.set_xlabel("Step Number")
            axes.set_label("Suboptimality")
            axes.set_title("Suboptimality Plot")
            axes.set_yscale('log')

            pdf.savefig(fig)
