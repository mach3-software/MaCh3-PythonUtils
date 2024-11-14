"""
ML Factory implementation, effectively a selector for making models
"""

from MaCh3PythonUtils.machine_learning.scikit_interface import SciKitInterface
from MaCh3PythonUtils.machine_learning.tf_manual_interface import TfManualInterface
from MaCh3PythonUtils.machine_learning.tf_autotune_interface import TfAutotuneInterface

import sklearn.ensemble as ske
import tensorflow.keras as tfk
from MaCh3PythonUtils.file_handling.chain_handler import ChainHandler


class MLFactory:
    # Implement algorithms here
    __IMPLEMENTED_ALGORITHMS = {
        "scikit" : {
            "randomforest"  : ske.RandomForestRegressor,
            "gradientboost" : ske.GradientBoostingRegressor,
            "adaboost"      : ske.AdaBoostRegressor,
            "histboost"     : ske.HistGradientBoostingRegressor
        },
        "tensorflow": {
            "sequential" : tfk.Sequential,
            "residual": tfk.Model,
            "autotune": None
        },
    }

    def __init__(self, input_chain: ChainHandler, prediction_variable: str, plot_name: str):
        """Constructor for ML factory method

        :param input_chain: ChainHandler instance
        :type input_chain: ChainHandler
        :param prediction_variable: Variable we want to predict the value of
        :type prediction_variable: str
        """        
        # Common chain across all instances of factory
        self._chain = input_chain
        self._prediction_variable = prediction_variable
        self._plot_name = plot_name
            

    def __setup_package_factory(self, package: str, algorithm: str, **kwargs):
        """Basic method for initialising factory method

        :param package: Name of package model comes from [i.e. SciKit, TensorFlow]
        :type package: str
        :param algorithm: Name of algorithm in the package [i.e. Sequential]
        :type algorithm: str
        :raises ValueError: Package not implemented
        :raises ValueError: Algorithm not Implemented
        :return: Model initialised with kwargs
        :rtype: Any
        """        
        
        package   = package.lower()
        if package not in self.__IMPLEMENTED_ALGORITHMS.keys():
            raise ValueError(f"{package} not included, currently accepted packages are :\n  \
                             {list(self.__IMPLEMENTED_ALGORITHMS.keys())}")
        
        algorithm = algorithm.lower()
        
        if algorithm not in self.__IMPLEMENTED_ALGORITHMS[package].keys():
            raise ValueError(f"{algorithm} not implemented for {package}, currently accepted algorithms for {package} are:\n \
                             {list(self.__IMPLEMENTED_ALGORITHMS[package].keys())}")

        return self.__IMPLEMENTED_ALGORITHMS[package][algorithm](**kwargs)

    def __make_scikit_model(self, algorithm: str, **kwargs)->SciKitInterface:
        """Generates scikit model instance

        :param algorithm: Algorithm from scikit
        :type algorithm: str
        :return: SciKitInterface wrapper around model
        :rtype: SciKitInterface
        """        
        # Simple wrapper for scikit packages
        interface = SciKitInterface(self._chain, self._prediction_variable, self._plot_name)
        interface.add_model(self.__setup_package_factory(package="scikit", algorithm=algorithm, **kwargs))

        return interface
    
    def __make_tensorflow_manual_model(self, algorithm: str,  **kwargs)->TfManualInterface:
        """Generates TensorFlow model interface

        :param algorithm: TensorFlow algorithm [NOT layers]
        :type algorithm: str
        :return: TfInterface wrapper around model
        :rtype: _type_
        """ 
        interface = TfManualInterface(self._chain, self._prediction_variable, self._plot_name)
        
        interface.add_model(self.__setup_package_factory(package="tensorflow", algorithm=algorithm))
        
        for layer in kwargs["Layers"]:
            layer_id = list(layer.keys())[0]
            
            interface.add_layer(layer_id, layer[layer_id])
        
        if algorithm == "residual":
            interface.build_model_residual(kwargs["BuildSettings"])
        else:
            interface.build_model_sequential(kwargs["BuildSettings"])
        
        interface.set_training_settings(kwargs["FitSettings"])

        return interface

    def __make_tensorflow_automated_model(self, **kwargs)->TfAutotuneInterface:
        interface = TfAutotuneInterface(self._chain, self._prediction_variable, self._plot_name)
        
        interface.build_auto_tuned_network(**kwargs)
        return interface
    
    
    def __make_tensorflow_model(self, algorithm: str, **kwargs):
        if algorithm=="autotune":
            return self.__make_tensorflow_automated_model(**kwargs)
        else:
            return self.__make_tensorflow_manual_model(algorithm, **kwargs)
    
    
    def make_interface(self, interface_type: str, algorithm: str, **kwargs):
        interface_type = interface_type.lower()
        match(interface_type):
            case "scikit":
                return self.__make_scikit_model(algorithm, **kwargs)
            case "tensorflow":
                return self.__make_tensorflow_model(algorithm, **kwargs)
            case _:
                raise Exception(f"{interface_type} not implemented!")