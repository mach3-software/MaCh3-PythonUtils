from typing import Any
from pandas import DataFrame

from file_handling.chain_handler import ChainHandler
from machine_learning.fml_interface import FmlInterface


class SciKitInterface(FmlInterface):
    def __init__(self, chain: ChainHandler, prediction_variable: str) -> None:
        super().__init__(chain, prediction_variable)
        
        self._model = None
    
    def train_model(self):
        print(f"Training Model")
        if self._model is None:
            raise ValueError("No Model has been set!")
        
        if self._test_data is None or self._test_labels is None:
            raise ValueError("No testing data set")
        
        self._model.fit(self._test_data, self._test_labels)
        
    def model_predict(self, testing_data: DataFrame):
        if self._model is None:
            raise ValueError("No Model has been set!")

        return self._model.predict(testing_data)
    
    def test_model(self):
        
        if self._model is None:
            raise ValueError("No Model has been set!")

        if self._training_data is None or self._training_labels is None:
            raise ValueError("No training data set")

        prediction = self.model_predict(self._training_data)
        train_as_numpy = self._training_labels.to_numpy().T[0]
        
        self.evaluate_model(prediction, train_as_numpy)
    
        