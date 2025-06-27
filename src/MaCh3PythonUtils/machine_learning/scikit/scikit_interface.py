from pandas import DataFrame
from MaCh3PythonUtils.machine_learning.file_ml_interface import FileMLInterface
from tqdm import tqdm
import sklearn.ensemble as ske
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV



"""
TODO: 
 - Add staged predict
"""

class SciKitInterface(FileMLInterface):   # SciKitInterface Class inherits the methods in FileMLInterface Class
                                          # SciKitInterface Class has additional methods which are used to train the model, FileML only has tests 
    def train_model(self):
        """Trains model

        :raises ValueError: Model not initialised
        :raises ValueError: Data set not initialised
        """        
        print(f"Training Model")
        scaled_data = self.scale_data(self._training_data)
        
        if self._model is None:
            raise ValueError("No Model has been set!")
        
        if self._training_data is None or self._training_labels is None:
            raise ValueError("No test data set")
        
        self._model.fit(scaled_data, self.scale_labels(self._training_labels))


        

        
    def model_predict(self, test_data: DataFrame)->list:
        """Gets model prediction

        :param test_data: Data to predict
        :type test_data: DataFrame
        :raises ValueError: No model set
        :return: Model prediction for test_data
        :rtype: list
        """        
        scale_data = self.scale_data(test_data)
        
        if self._model is None:
            raise ValueError("No Model has been set!")

        return self._model.predict(scale_data)
    

