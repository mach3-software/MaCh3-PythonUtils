from abc import ABC, abstractmethod
from file_handling.chain_handler import ChainHandler
from typing import Any, Tuple
from sklearn.model_selection import train_test_split
import pandas as pd

class FmlInterface(ABC):
    """Abstract interface with file handler which should be used for ML models
    """
    
    def __init__(self, chain: ChainHandler, prediction_variable: str) -> None:
        self._chain = chain
        
        self._prediction_variable = prediction_variable
        
        if prediction_variable not in self._chain.ttree_array.columns:
            raise ValueError(f"Cannot find {prediction_variable} in input tree")
        
        self._model = None
        
        self._training_data=None
        self._training_labels=None
        self._test_data=None
        self._test_labels=None
            
    def separate_dataframe(self)->Tuple[pd.DataFrame, pd.DataFrame]:
        features = self._chain.ttree_array.copy()
        labels   = pd.DataFrame(features.pop(self._prediction_variable) )
        
        return features, labels
    
    def set_training_test_set(self, features, labels, test_size: float):
        self._training_data, self._test_data, self._training_labels, self._test_labels =  train_test_split(features, labels, test_size=test_size)

    @property
    def model(self)->Any:
        return self._model    
    
    def add_model(self, ml_model: Any)->None:
        # Add ML model into your interface
        self._model = ml_model
    
    @abstractmethod
    def train_model(self):
        pass
    
    
    @abstractmethod
    def model_predict(self, testing_data):
        pass
    
    @abstractmethod
    def test_model(self):
        pass
    
    @abstractmethod
    def evaluate_model():
        pass
