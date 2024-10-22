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
            # Name of chain in file
            "ChainName": "",
            # More printouts
            "Verbose": False,
            # Make posteriors from chain?
            "MakePosteriors": False,
            # Run Diagnostics code?
            "MakeDiagnostics": False,
            # Make an ML model to replicate the chain likelihood model
            "MakeMLModel": False,
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
        
        # Settings for plotting tools
        "PlottingSettings":{
        # Specific Settings for posterior plots
            "PosteriorSettings": {
                # Make 2D Posterior Plots?
                "Make2DPosteriors": False,
                # Make a triangle plot
                "MakeTrianglePlot": False,
                # Variables in the triangle plot
                "TrianglePlot": [],
                # 1D credible intervals
                "MakeCredibleIntervals": False,
                # Output file
                "PosteriorOutputFile": "posteriors.pdf"
            },
            
            # Specific Settings for diagnostic plots
            "DiagnosticsSettings": {
                # Make violin plot?
                "MakeViolin": False,
                # Make trace + AC plot?
                "MakeTraceAC": False,
                # Make effective sample size plot?
                "MakeESS": False,
                # Make MCSE plot?
                "MakeMCSE": False,
                # Make suboptimality plot
                "MakeSuboptimality": False,
                # Step for calculation
                "SuboptimalitySteps": 0,
                # Output file
                "DiagnosticsOutputFile": "diagnostics.pdf",
                # Print summary statistic
                "PrintSummary": False
        }
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
            "NSteps": 100000
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

    
    
    def make_posterior_plots(self):
        """Generates posterior plots
        """        
        if self._plot_interface is None:
            self._plot_interface = PlottingInterface(self._file_handler)
        
        posterior_labels = []
        
        if self.__chain_settings['PlottingSettings']['PosteriorSettings']['Make2DPosteriors']:
            self._plot_interface.initialise_new_plotter(PosteriorPlotter2D(self._file_handler), 'posterior_2d')
            posterior_labels.append('posterior_2d')

        if self.__chain_settings['PlottingSettings']['PosteriorSettings']['MakeTrianglePlot']:
            self._plot_interface.initialise_new_plotter(TrianglePlotter(self._file_handler), 'posterior_triangle')
            posterior_labels.append('posterior_triangle')
        
        # Which variables do we actually want 2D plots for?
        self._plot_interface.set_variables_to_plot(self.__chain_settings['PlottingSettings']['PosteriorSettings']['TrianglePlot'], posterior_labels)

        if self.__chain_settings['PlottingSettings']['PosteriorSettings']['Make1DPosteriors']:
            self._plot_interface.initialise_new_plotter(PosteriorPlotter1D(self._file_handler), 'posterior_1d')
            posterior_labels.append('posterior_1d')
    
    
        self._plot_interface.set_credible_intervals(self.__chain_settings['PlottingSettings']['PosteriorSettings']['CredibleIntervals'])
        self._plot_interface.set_is_circular(self.__chain_settings['ParameterSettings']['CircularParameters'])

        self._plot_interface.make_plots(self.__chain_settings['PlottingSettings']['PosteriorSettings']['PosteriorOutputFile'], posterior_labels)

    def make_diagnostics_plots(self):
        """Generates diagnostics plots
        """        
        if self._plot_interface is None:
            self._plot_interface = PlottingInterface(self._file_handler)

        diagnostic_labels = []

        if self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['MakeViolin']:
            self._plot_interface.initialise_new_plotter(ViolinPlotter(self._file_handler), 'violin_plotter')
            diagnostic_labels.append('violin_plotter')
        if self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['MakeTraceAC']:
            self._plot_interface.initialise_new_plotter(AutocorrelationTracePlotter(self._file_handler), 'trace_autocorr')
            diagnostic_labels.append('trace_autocorr')
        if self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['MakeESS']:
            self._plot_interface.initialise_new_plotter(EffectiveSampleSizePlotter(self._file_handler), 'ess_plot')
            diagnostic_labels.append('ess_plot')
        if self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['MakeMCSE']:
            self._plot_interface.initialise_new_plotter(MarkovChainStandardError(self._file_handler), 'msce_plot')
            diagnostic_labels.append('msce_plot')

        self._plot_interface.make_plots(self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['DiagnosticsOutputFile'], diagnostic_labels)

        # Final one, covariance plotter
        if self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['MakeSuboptimality']:
            suboptimality_obj = CovarianceMatrixUtils(self._file_handler)
            suboptimality_obj.calculate_suboptimality(self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['SuboptimalitySteps'], self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['SubOptimalityMin'])
            suboptimality_obj.plot_suboptimality(f"suboptimality_{self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['DiagnosticsOutputFile']}")

        # Finally let's make a quick simmary
        if self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['PrintSummary']:
            self._plot_interface.print_summary(f"summary_{self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['DiagnosticsOutputFile']}.txt")

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
            self._interface.load_model(external_model)
        
        else:
            self._interface.set_training_test_set(self.__chain_settings["MLSettings"]["TestSize"])
            self._interface.train_model()
            self._interface.test_model()
            self._interface.save_model(self.__chain_settings["MLSettings"]["MLOutputFile"])

    
    def __call__(self) -> None:
        """Runs over all files from config
        """        
        self.make_file_handler()
        if self.__chain_settings["FileSettings"]["MakePosteriors"]:
            self.make_posterior_plots()
        
        if self.__chain_settings["FileSettings"]["MakeDiagnostics"]:
            self.make_diagnostics_plots()
        
        if self.__chain_settings["FileSettings"]["MakeMLModel"]:
            self.make_ml_interface()

            if self.__chain_settings["FileSettings"]["RunLLHScan"] and self._interface is not None:
                self._interface.run_likelihood_scan(self.__chain_settings["LikelihoodScanSettings"]["NDivisions"])
                
            if self.__chain_settings["FileSettings"]["RunMCMC"] and self._interface is not None:

                mcmc = MCMCMultGPU(self._interface,
                        self.__chain_settings["MCMCSettings"]["NChains"],
                        self.__chain_settings["ParameterSettings"]["CircularParameters"],
                        self.__chain_settings["MCMCSettings"]["UpdateStep"])

                mcmc(self.__chain_settings["MCMCSettings"]["NSteps"],
                     self.__chain_settings["MCMCSettings"]["MCMCOutput"])

                

