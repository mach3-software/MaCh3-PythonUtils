"""
ML Factory implementation, effectively a selector for making models
"""

from MaCh3PythonUtils.machine_learning.scikit.scikit_interface import SciKitInterface
from MaCh3PythonUtils.machine_learning.tensorflow.tf_autotune_interface import TfAutotuneInterface
from MaCh3PythonUtils.machine_learning.tensorflow.tf_sequential_model import TfSequentialModel
from MaCh3PythonUtils.machine_learning.tensorflow.tf_residual_model import TfResidualModel
from MaCh3PythonUtils.machine_learning.tensorflow.tf_normalizing_flow_model import TfNormalizingFlowModel

from MaCh3PythonUtils.machine_learning.tensorflow.tf_manual_interface import TfManualLayeredInterface
from MaCh3PythonUtils.machine_learning.tensorflow.tf_interface import TfInterface

from MaCh3PythonUtils.file_handling.chain_handler import ChainHandler

import sklearn.ensemble as ske
import tensorflow.keras as tfk


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
            "sequential" : TfSequentialModel,
            "residual": TfResidualModel,
            "normalizing_flow": TfNormalizingFlowModel,
            "autotune": TfAutotuneInterface
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
        
    
    def __make_tensorflow_layered_model(self, interface: TfManualLayeredInterface, layers: dict)->TfManualLayeredInterface:
        for layer in layers:
            layer_id = list(layer.keys())[0]                
            interface.add_layer(layer_id, layer[layer_id].copy())

        return interface

    def __make_tensorflow_model(self, algorithm: str, **kwargs)->TfInterface:
        model_func = self.__IMPLEMENTED_ALGORITHMS["tensorflow"].get(algorithm.lower(), None)
        
        if model_func is None:
            raise Exception(f"Cannot find {algorithm}")
        
        model: TfInterface = model_func(self._chain, self._prediction_variable, self._plot_name)

        # Ugh
        if algorithm=="sequential" or algorithm=="residual":
            model = self.__make_tensorflow_layered_model(model, kwargs["Layers"])
            model.set_training_settings(kwargs.get("FitSettings"))


        model.build_model(**kwargs["BuildSettings"])
        
        return model

    def make_interface(self, interface_type: str, algorithm: str, **kwargs):
        interface_type = interface_type.lower()
        match(interface_type):
            case "scikit":
                return self.__make_scikit_model(algorithm, **kwargs)
            case "tensorflow":
                return self.__make_tensorflow_model(algorithm, **kwargs)
            case _:
                raise Exception(f"{interface_type} not implemented!")