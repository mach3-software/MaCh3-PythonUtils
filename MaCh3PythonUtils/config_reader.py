import yaml

from file_handling.chain_handler import ChainHandler
from machine_learning.ml_factory import MLFactory
from machine_learning.fml_interface import FmlInterface


class ConfigReader:    
    
    # Strictly unecessary but nice conceptually
    _file_handler = None
    _interface    = None
    
    def __init__(self, config: str):
        with open(config, 'r') as c:
            self._yaml_config = yaml.safe_load(c)    
    
    
    def setup_file_handler(self)->None:
        # Process MCMC chain    
        self._file_handler = ChainHandler(self._yaml_config["FileSettings"]["FileName"],
                                    self._yaml_config["FileSettings"]["ChainName"],
                                    self._yaml_config["FileSettings"]["Verbose"])
        
        self._file_handler.ignore_plots(self._yaml_config["FileSettings"]["IgnoredParameters"])
        self._file_handler.add_additional_plots(self._yaml_config["FileSettings"]["ParameterNames"])
        self._file_handler.add_additional_plots(self._yaml_config["FileSettings"]["LabelName"], True)

        self._file_handler.add_new_cuts(self._yaml_config["FileSettings"]["ParameterCuts"])

        self._file_handler.convert_ttree_to_array()

    
    def setup_ml_interface(self)->None:
        if self._file_handler is None:
            raise Exception("Cannot initialise ML interface without first setting up file handler!")
        
        
        factory = MLFactory(self._file_handler, self._yaml_config["FileSettings"]["LabelName"])
        if self._yaml_config["FitterSettings"]["FitterPackage"].lower() == "scikit":        
            self._interface = factory.setup_scikit_model(self._yaml_config["FitterSettings"]["FitterName"],
                                    **self._yaml_config["FitterSettings"]["FitterKwargs"])

        elif self._yaml_config["FitterSettings"]["FitterPackage"].lower() == "tensorflow":        
            self._interface = factory.setup_tensorflow_model(self._yaml_config["FitterSettings"]["FitterName"],
                                    **self._yaml_config["FitterSettings"]["FitterKwargs"])

        else:
            raise ValueError("Input not recognised!")
        
        if self._yaml_config["FitterSettings"].get("AddFromExternalModel"):
            external_model = self._yaml_config["FitterSettings"]["ExternalModel"]
            self._interface.load_model(external_model)
        
        else:
            self._interface.set_training_test_set(self._yaml_config["FitterSettings"]["TestSize"])
        
            self._interface.train_model()
            self._interface.test_model()
            self._interface.save_model(self._yaml_config["FileSettings"]["ModelOutputName"])
        
    def __call__(self) -> None:
        self.setup_file_handler()
        self.setup_ml_interface()

        
    @property
    def chain_handler(self)->ChainHandler | None:
        return self._file_handler
    
    @property
    def ml_interface(self)->FmlInterface | None:
        return self._interface