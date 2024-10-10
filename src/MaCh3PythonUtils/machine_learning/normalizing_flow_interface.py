'''
Implementation of https://pypi.org/project/normflows/ pacakage
'''

import torch
import pandas as pd

from MaCh3PythonUtils.machine_learning.file_ml_interface import FileMLInterface


class NormalisingFlowInterface(FileMLInterface):
    def train_model(self):
        # Move to GPU if possible
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = self._model.to(device)
        self._model = self._model.double()
        # Fit model
        self._model.fit(self._training_data, self._training_labels)
        
    def model_predict(self, test_data: pd.DataFrame)->None:
        return self._model.predict(test_data)
    
    