import yaml

from MaCh3PythonUtils.file_handling.chain_handler import ChainHandler
from MaCh3PythonUtils.machine_learning.ml_factory import MLFactory
from MaCh3PythonUtils.diagnostics.interface.plotting_interface import PlottingInterface
from MaCh3PythonUtils.diagnostics.mcmc_plots.posteriors.posteriors_1d import PosteriorPlotter1D
from MaCh3PythonUtils.diagnostics.mcmc_plots.posteriors.posteriors_2d import PosteriorPlotter2D, TrianglePlotter
from MaCh3PythonUtils.diagnostics.mcmc_plots.diagnostics.covariance_matrix_utils import CovarianceMatrixUtils
from MaCh3PythonUtils.diagnostics.mcmc_plots.diagnostics.autocorrelation_trace_plotter import AutocorrelationTracePlotter

from MaCh3PythonUtils.diagnostics.mcmc_plots.diagnostics.simple_diag_plots import EffectiveSampleSizePlotter,\
                                                                                     MarkovChainStandardError, ViolinPlotter

from MaCh3PythonUtils.fitters.multi_mcmc_gpu import MCMCMultGPU

from deepmerge import always_merger

class ConfigReader:
    # Strictly unecessary but nice conceptually
    _file_handler = None
    _interface    = None
    _plot_interface = None
    
    __default_settings = {
        # Settings for file and I/O
        "FileSettings" : {
            # Name of input file
            "FileName": "",
            # Skip loading a file? Useful for ML
            "SkipFileLoading": False,
            # Name of chain in file
            "ChainName": "",
            # More printouts
            "Verbose": False,
            # Run an LLH Scan?
            "RunLLHScan": False,
            # Run MCMC?
            "RunMCMC": False
        },
        
        "ParameterSettings":{
            "CircularParameters" : [],
            # List of parameter names
            "ParameterNames":[],
            # List of cuts
            "ParameterCuts":[],
            # Name of label branch, used in ML
            "LabelName": "",
            # Any parameters we want to ignore
            "IgnoredParameters":[]
        },
        
        # Specific Settings for ML Applications
        "MLSettings": {
            # Name of plots
            "PlotOutputName": "ml_output.pdf",
            # Fitter package either SciKit or TensorFlow
            "FitterPackage": "",
            # Fitter Model
            "FitterName": "",
            # Keyword arguments for fitter
            "FitterKwargs" : {},
            #Use an external model that's already been trained?
            "AddFromExternalModel": False,
            # Proportion of input data set used for testing (range of 0-1 )
            "TestSize": 0.0,
            # Name to save ML model in
            "MLOutputFile": "mlmodel.pkl"
        },

        # Settings for LLH Scan
        "LikelihoodScanSettings": {
            "NDivisions": 100
        },
            

        # Settings for MCMC
        "MCMCSettings": {
            "NSteps": 100000,
            "NChains": 1,
            "UpdateStep": 100,
            "MaxUpdateSteps": 500000
        }
    }
    
    
    
    def __init__(self, config: str):
        """Constructor

        :param config: Name of yaml config
        :type config: str
        """        
        with open(config, 'r') as c:
            yaml_config = yaml.safe_load(c)    

        # Update default settings
        self.__chain_settings = always_merger.merge(self.__default_settings, yaml_config)
    
    def make_file_handler(self)->None:
        """Sets up file handler object
        """        
        # Process MCMC chain    
        self._file_handler = ChainHandler(self.__chain_settings["FileSettings"]["FileName"],
                                    self.__chain_settings["FileSettings"]["ChainName"],
                                    self.__chain_settings["FileSettings"]["Verbose"])
        
        self._file_handler.ignore_plots(self.__chain_settings["ParameterSettings"]["IgnoredParameters"])
        self._file_handler.add_additional_plots(self.__chain_settings["ParameterSettings"]["ParameterNames"])
        
        self._file_handler.add_additional_plots(self.__chain_settings["ParameterSettings"]["LabelName"], True)

        self._file_handler.add_new_cuts(self.__chain_settings["ParameterSettings"]["ParameterCuts"])

        self._file_handler.convert_ttree_to_array()

    
    def make_ml_interface(self)->None:
        """Generates ML interface objects
        """        
        if self._file_handler is None:
            raise Exception("Cannot make interface without opening a file!")
        
        factory = MLFactory(self._file_handler, self.__chain_settings["ParameterSettings"]["LabelName"], self.__chain_settings["MLSettings"]["PlotOutputName"])

        self._interface = factory.make_interface(self.__chain_settings["MLSettings"]["FitterPackage"],
                                                 self.__chain_settings["MLSettings"]["FitterName"],
                                                 **self.__chain_settings["MLSettings"]["FitterKwargs"])
  
        if self.__chain_settings["MLSettings"].get("AddFromExternalModel"):
            external_model = self.__chain_settings["MLSettings"]["ExternalModel"]
            external_scaler = self.__chain_settings["MLSettings"]["ExternalScaler"]
            self._interface.load_model(external_model)
            self._interface.load_scaler(external_scaler)
        
        else:
            self._interface.set_training_test_set(self.__chain_settings["MLSettings"]["TestSize"])
            self._interface.train_model()
            self._interface.test_model()
            self._interface.save_model(self.__chain_settings["MLSettings"]["MLOutputFile"])
            self._interface.save_scaler(self.__chain_settings['MLSettings']['MLScalerOutputName'])
    
    def run_mcmc(self):
        print("WARNING: MCMC HAS ONLY BEEN TESTED WITH TENSORFLOW INTERFACES!")
        mcmc = MCMCMultGPU(self._interface,
                self.__chain_settings["MCMCSettings"]["NChains"],
                self.__chain_settings["ParameterSettings"]["CircularParameters"],
                self.__chain_settings["MCMCSettings"]["UpdateStep"],
                self.__chain_settings["MCMCSettings"]["MaxUpdateSteps"])

        mcmc(self.__chain_settings["MCMCSettings"]["NSteps"],
                self.__chain_settings["MCMCSettings"]["MCMCOutput"],)

    
    def __call__(self) -> None:
        """Runs over all files from config
        """         
        self.make_file_handler()
            
        self.make_ml_interface()

        if self.__chain_settings["FileSettings"]["RunLLHScan"] and self._interface is not None:
            self._interface.run_likelihood_scan(self.__chain_settings["LikelihoodScanSettings"]["NDivisions"])
            
        if self.__chain_settings["FileSettings"]["RunMCMC"] and self._interface is not None:
            self.run_mcmc()
                

