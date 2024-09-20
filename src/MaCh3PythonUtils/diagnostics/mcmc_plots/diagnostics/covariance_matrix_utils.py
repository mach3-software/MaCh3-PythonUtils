'''
Additional Diagnostics that can be used with MCMC but don't rely on plotting

Suboptimality : https://www.jstor.org/stable/25651249?seq=3
'''

from MaCh3PythonUtils.file_handling.chain_handler import ChainHandler
import numpy as np
from scipy.linalg import sqrtm
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import mplhep as hep
from concurrent.futures import ThreadPoolExecutor, as_completed


class CovarianceMatrixUtils:
    def __init__(self, chain_handler: ChainHandler)->None:
        """Constructor for handling covariance matrix

        :param chain_handler: Instance of a chain handler objec
        :type chain_handler: ChainHandler
        """
        # Let's just ignore some warnings :grin:
        warnings.filterwarnings("ignore", category=DeprecationWarning) #Some imports are a little older
        warnings.filterwarnings("ignore", category=UserWarning) #Some imports are a little older

        chain_handler.get_ttree_as_arviz() # Ensure that we have things in the correct format
        self._parameter_names = list(chain_handler.arviz_tree.keys())
        self._total_parameters = len(self._parameter_names)
        self._ttree_data_frame = chain_handler.arviz_tree.to_dataframe() # All our handling will be done using thi s object
        # Various useful class properties
        self._suboptimality_array = [] # For filling with suboptimality
        self._suboptimality_evaluation_points = [] #Where did we evalutate this?
        self._full_covariance = self.calculate_covariance_matrix()
        self._sqrt_full_covariance_inv = np.linalg.inv(sqrtm(self._full_covariance))
        plt.style.use(hep.style.ROOT)

    def calculate_covariance_matrix(self, min_step: int=0, max_step: int=-1)->np.ndarray:
        """_summary_

        :param min_step: Steps at which to calculate covariance matrix at, defaults to 0
        :type min_step: int, optional
        :param max_step: Maximum step to calculate covariance matrix at, defaults to -1 indicating you want to evaluate the full chain
        :type max_step: int, optional
        :return: Covariance matrix at each Nth step
        :rtype: np.ndarray
        """
        if(max_step<=0):
            sub_array = self._ttree_data_frame[min_step : ]
        else:
            sub_array = self._ttree_data_frame[min_step : max_step]


        return sub_array.cov().to_numpy()

    def _calculate_matrix_suboptimality(self, step_number: int)->float:
        """Calculate the suboptimality value for the mtarix for step N

        :param step_number: calculate the covariance matrix for the first step_number steps
        :type step_number: int
        :return: Suboptimality
        :rtype: float
        """
        new_covariance = self.calculate_covariance_matrix(max_step=step_number)

        sqrt_input_cov = sqrtm(new_covariance)

        # Get product of square roots
        matrix_prod = sqrt_input_cov @ self._sqrt_full_covariance_inv
        
        #Get eigen values 
        eigenvalues, _ = np.linalg.eig(matrix_prod)
        
        return self._total_parameters * np.sum(eigenvalues**(-2))/((np.sum(eigenvalues)**(-1))**2)

    def calculate_suboptimality(self, step_skip : int = 1000, min_step=0)->None:
        """
        Calculates the suboptimality for every step_skip steps

        :param step_skip: Number of steps to skip, defaults to 1000
        :type step_skip: int, optional
        :param min_step: smallest number of starting steps, defaults to 0
        :type min_step: int, optional
        """
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
        """Plots suboptimality value

        :param output_file: Output PDF file
        :type output_file: str
        """        
        print(f"Saving to {output_file}")
        with PdfPages(output_file) as pdf:
            fig, axes = plt.subplots()

            axes.plot(self._suboptimality_evaluation_points, self._suboptimality_array)
            axes.set_xlabel("Step Number")
            axes.set_label("Suboptimality")
            axes.set_title("Suboptimality Plot")
            axes.set_yscale('log')

            pdf.savefig(fig)
