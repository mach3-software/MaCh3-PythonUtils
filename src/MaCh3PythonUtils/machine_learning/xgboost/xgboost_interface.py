from pandas import DataFrame
from MaCh3PythonUtils.machine_learning.file_ml_interface import FileMLInterface
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

"""
XGBoost interface for MaCh3PythonUtils
Provides a wrapper around XGBoost models following the FileMLInterface pattern
"""

from xgboost import XGBRegressor, XGBClassifier, XGBRFRegressor, XGBRFClassifier

class XGBoostInterface(FileMLInterface):
    _model: XGBRegressor | XGBClassifier | XGBRFRegressor | XGBRFClassifier    
    
    
    def train_model(self):
        """Trains XGBoost model with progress monitoring

        :param show_progress: Whether to show training progress bar, defaults to True
        :type show_progress: bool
        :param validation_split: Fraction of training data to use for validation, defaults to 0.2
        :type validation_split: float
        :raises ValueError: Model not initialised
        :raises ValueError: Data set not initialised
        """        
        if self._model is None:
            raise ValueError("No Model has been set!")
        
        if self.scaled_train_data is None or self.scaled_train_labels is None:
            raise ValueError("No test data set")
        
        # Convert labels to 1D array if needed (XGBoost expects 1D labels)
        labels = self.scaled_train_labels
        if hasattr(labels, 'values'):
            labels = labels.values
        if hasattr(labels, 'shape') and len(labels.shape) > 1:
            labels = np.ravel(labels)
        
        # Need to set up evals
        
        train_indices, val_indices = train_test_split(
            np.arange(len(self.scaled_train_data)), test_size=0.2, shuffle=True
        )

        self._model.fit(self.scaled_train_data[train_indices], self.scaled_train_labels[train_indices],
                        # eval_set=[(self.scaled_train_data[val_indices], self.scaled_train_labels[val_indices])],
                        verbose=False)

    def model_predict(self, test_data: DataFrame) -> np.ndarray:
        """Gets model prediction

        :param test_data: Data to predict
        :type test_data: DataFrame
        :raises ValueError: No model set
        :return: Model prediction for test_data
        :rtype: np.ndarray
        """        
        scale_data = self.scale_data(test_data)
        
        if self._model is None:
            raise ValueError("No Model has been set!")

        predictions = self._model.predict(scale_data)
        
        return predictions
            
