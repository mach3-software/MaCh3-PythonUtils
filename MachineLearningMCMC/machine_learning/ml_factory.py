"""
ML Factory implementation, effectively a selector for making models
"""

from typing import Any, Dict

from machine_learning.scikit_interface import SciKitInterface
import sklearn.ensemble as ske

from file_handling.chain_handler import ChainHandler
from functools import partial



class MLFactory:
    # Implement algorithms here
    __IMPLEMENTED_ALGORITHMS = {
        "scikit" : {
            "randomforest"  : ske.RandomForestRegressor,
            "gradientboost" : ske.GradientBoostingRegressor,
            "adaboost"      : ske.AdaBoostRegressor
        },
        "tensorflow":
            {      
            }
    }

    def __init__(self, input_chain: ChainHandler, prediction_variable: str):
        # Common chain across all instances of factory
        self._chain = input_chain
        self._prediction_variable = prediction_variable
        
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
    

    def __setup_package_factory(self, package: str, algorithm: str, **kwargs):
        """
        Rough method for setting up a package
        """
        
        package   = package.lower()
        if package not in self.__IMPLEMENTED_ALGORITHMS:
            raise ValueError(f"{package} not included, currently accepted packages are :\n  \
                             {list(self.__IMPLEMENTED_ALGORITHMS.keys())}")
        
        algorithm = algorithm.lower()
        
        if algorithm not in self.__IMPLEMENTED_ALGORITHMS[package]:
            raise ValueError(f"{algorithm} not implemented for {package}, currently accepted algorithms for {package} are:\n \
                             {list(self.__IMPLEMENTED_ALGORITHMS[package].keys())}")
            
        return self.__IMPLEMENTED_ALGORITHMS[package][algorithm](**kwargs)

    def setup_scikit_model(self, algorithm: str, **kwargs)->SciKitInterface:
        # Simple wrapper for scikit packages
        interface = SciKitInterface(self._chain, self._prediction_variable)
        interface.add_model(self.__setup_package_factory(package="scikit", algorithm=algorithm, **kwargs))
        return interface
    
    def setup_tensorflow_model(self, algorithm: str, network_structure: Dict[str, Any], **kwargs):
        model = self.__setup_package_factory(package="tesnorflow", algorithm=algorithm, kwargs=kwargs)
        return model
    