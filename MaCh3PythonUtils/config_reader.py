import yaml

from file_handling.chain_handler import ChainHandler
from machine_learning.ml_factory import MLFactory
from machine_learning.fml_interface import FmlInterface
from diagnostics.interface.plotting_interface import PlottingInterface
import diagnostics.plotters.posteriors as m3post
import diagnostics.plotters.diagnostics as m3diag
from pydantic.utils import deep_update

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
            "MakeMLModel": False
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
            "MLOutputFile": "mlmodel"
        }
        
    }
    
    
    
    def __init__(self, config: str):
        with open(config, 'r') as c:
            self._yaml_config = yaml.safe_load(c)    

        # Update default settings
        self.__chain_settings = deep_update(self.__default_settings, self._yaml_config)
    
    def make_file_handler(self)->None:
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
        if self._plot_interface is None:
            self._plot_interface = PlottingInterface(self._file_handler)
        
        posterior_labels = []
        
        if self.__chain_settings['PlottingSettings']['PosteriorSettings']['Make2DPosteriors']:
            self._plot_interface.initialise_new_plotter(m3post.PosteriorPlotter2D(self._file_handler), 'posterior_2d')
            posterior_labels.append('posterior_2d')

        if self.__chain_settings['PlottingSettings']['PosteriorSettings']['MakeTrianglePlot']:
            self._plot_interface.initialise_new_plotter(m3post.TrianglePlotter(self._file_handler), 'posterior_triangle')
            posterior_labels.append('posterior_triangle')
        
        # Which variables do we actually want 2D plots for?
        self._plot_interface.set_variables_to_plot(self.__chain_settings['PlottingSettings']['PosteriorSettings']['TrianglePlot'], posterior_labels)

        if self.__chain_settings['PlottingSettings']['PosteriorSettings']['Make1DPosteriors']:
            self._plot_interface.initialise_new_plotter(m3post.PosteriorPlotter1D(self._file_handler), 'posterior_1d')
            posterior_labels.append('posterior_1d')
    
    
        self._plot_interface.set_credible_intervals(self.__chain_settings['PlottingSettings']['PosteriorSettings']['CredibleIntervals'])
        self._plot_interface.set_is_circular(self.__chain_settings['ParameterSettings']['CircularParameters'])

        self._plot_interface.make_plots(self.__chain_settings['PlottingSettings']['PosteriorSettings']['PosteriorOutputFile'], posterior_labels)

    def make_diagnostics_plots(self):
        if self._plot_interface is None:
            self._plot_interface = PlottingInterface(self._file_handler)

        diagnostic_labels = []

        if self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['MakeViolin']:
            self._plotting_interface.initialise_new_plotter(m3diag.ViolinPlotter(self._file_handler), 'violin_plotter')
            diagnostic_labels.append('violin_plotter')
        if self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['MakeTraceAC']:
            self._plotting_interface.initialise_new_plotter(m3diag.AutocorrelationTracePlotter(self._file_handler), 'trace_autocorr')
            diagnostic_labels.append('trace_autocorr')
        if self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['MakeESS']:
            self._plotting_interface.initialise_new_plotter(m3diag.EffectiveSampleSizePlotter(self._file_handler), 'ess_plot')
            diagnostic_labels.append('ess_plot')
        if self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['MakeMCSE']:
            self._plotting_interface.initialise_new_plotter(m3diag.MarkovChainStandardError(self._file_handler), 'msce_plot')
            diagnostic_labels.append('msce_plot')

        self._plotting_interface.make_plots(self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['DiagnosticsOutputFile'], diagnostic_labels)

        # Final one, covariance plotter
        if self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['MakeSuboptimality']:
            suboptimality_obj = m3diag.CovarianceMatrixUtils(self._file_handler)
            suboptimality_obj.calculate_suboptimality(self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['SuboptimalitySteps'], self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['SubOptimalityMin'])
            suboptimality_obj.plot_suboptimality(f"suboptimality_{self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['DiagnosticsOutputFile']}")

        # Finally let's make a quick simmary
        if self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['PrintSummary']:
            self._plotting_interface.print_summary(f"summary_{self.__chain_settings['PlottingSettings']['DiagnosticsSettings']['DiagnosticsOutputFile']}.txt")

    def make_ml_interface(self)->None:
        if self._file_handler is None:
            raise Exception("Cannot initialise ML interface without first setting up file handler!")
        
        
        factory = MLFactory(self._file_handler, self.__chain_settings["ParameterSettings"]["LabelName"])
        if self.__chain_settings["MLSettings"]["FitterPackage"].lower() == "scikit":        
            self._interface = factory.make_scikit_model(self.__chain_settings["MLSettings"]["FitterName"],
                                    **self.__chain_settings["MLSettings"]["FitterKwargs"])

        elif self.__chain_settings["MLSettings"]["FitterPackage"].lower() == "tensorflow":        
            self._interface = factory.make_tensorflow_model(self.__chain_settings["MLSettings"]["FitterName"],
                                    **self.__chain_settings["MLSettings"]["FitterKwargs"])

        else:
            raise ValueError("Input not recognised!")
        
        if self.__chain_settings["MLSettings"].get("AddFromExternalModel"):
            external_model = self.__chain_settings["MLSettings"]["ExternalModel"]
            self._interface.load_model(external_model)
        
        else:
            self._interface.set_training_test_set(self.__chain_settings["MLSettings"]["TestSize"])
        
            self._interface.train_model()
            self._interface.test_model()
            self._interface.save_model(self.__chain_settings["MLSettings"]["MLOutputFile"])

    
    
    def __call__(self) -> None:
        
        self.make_file_handler()
        if self.__chain_settings["FileSettings"]["MakePosteriors"]:
            self.make_posterior_plots()
        
        if self.__chain_settings["FileSettings"]["MakeDiagnostics"]:
            self.make_diagnostics_plots()
        
        if self.__chain_settings["FileSettings"]["MakeMLModel"]:
            self.make_ml_interface()

    
    @property
    def chain_handler(self)->ChainHandler | None:
        return self._file_handler
    
    @property
    def ml_interface(self)->FmlInterface | None:
        return self._interface
    
