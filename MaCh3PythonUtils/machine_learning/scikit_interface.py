from typing import Any
from pandas import DataFrame

from file_handling.chain_handler import ChainHandler
from machine_learning.fml_interface import FmlInterface

"""
TODO: 
 - Add staged predict
"""

class SciKitInterface(FmlInterface):
    def __init__(self, chain: ChainHandler, prediction_variable: str) -> None:
        super().__init__(chain, prediction_variable)
        
        self._model = None
    
    def train_model(self):
        print(f"Training Model")
        if self._model is None:
            raise ValueError("No Model has been set!")
        
        if self._training_data is None or self._training_labels is None:
            raise ValueError("No test data set")
        
        self._model.fit(self._training_data, self._training_labels)
        
    def model_predict(self, test_data: DataFrame):
        
        scale_data = self._scalar.transform(test_data)
        
        if self._model is None:
            raise ValueError("No Model has been set!")

        return self._model.predict(scale_data)
    