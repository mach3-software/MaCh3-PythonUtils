from pandas import DataFrame
from MaCh3PythonUtils.machine_learning.file_ml_interface import FileMLInterface

"""
TODO: 
 - Add staged predict
"""

class SciKitInterface(FileMLInterface):    
    def train_model(self):
        """Trains model

        :raises ValueError: Model not initialised
        :raises ValueError: Data set not initialised
        """        
        print(f"Training Model")
        if self._model is None:
            raise ValueError("No Model has been set!")
        
        if self._training_data is None or self._training_labels is None:
            raise ValueError("No test data set")
        
        self._model.fit(self._training_data, self._training_labels)
        
    def model_predict(self, test_data: DataFrame)->list:
        """Gets model prediction

        :param test_data: Data to predict
        :type test_data: DataFrame
        :raises ValueError: No model set
        :return: Model prediction for test_data
        :rtype: list
        """        
        scale_data = self._scalar.transform(test_data)
        
        if self._model is None:
            raise ValueError("No Model has been set!")

        return self._model.predict(scale_data)
    