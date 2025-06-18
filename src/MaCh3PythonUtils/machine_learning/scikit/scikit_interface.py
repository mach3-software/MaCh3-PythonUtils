from pandas import DataFrame
from MaCh3PythonUtils.machine_learning.file_ml_interface import FileMLInterface
from tqdm import tqdm
import sklearn.ensemble as ske

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
    '''
    def test_model_class(self):

        model = ske.HistGradientBoostingClassifier(max_bins=255, max_iter=100) 
        model.fit(self.test_data, self._training_labels) #trains the model to fit the test data with the training labels 
        y_pred = model.predict(self.test_data) # "model, can you predict training labels from test data"

        return y_pred    
    
    '''