from typing import Any
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor

from file_handling.chain_handler import ChainHandler
from machine_learning.fml_interface import FmlInterface

from sklearn import metrics
import matplotlib.pyplot as plt

class RandomForestInterface(FmlInterface):
    def __init__(self, chain: ChainHandler, prediction_variable: str) -> None:
        super().__init__(chain, prediction_variable)
        
        self._model: RandomForestRegressor | None = None
    
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
        
        print(prediction)
        print(train_as_numpy)
        
        print(f"Score : {metrics.mean_absolute_error(prediction,train_as_numpy)}")
        
        
        plt.plot(100*(prediction-train_as_numpy)/(0.5*(train_as_numpy+prediction)), color='r', label="Percentage Difference", linewidth=0.1)
        plt.legend()
        plt.savefig("DummyTest.pdf")
        
    def evaluate_model(self):
        return 0