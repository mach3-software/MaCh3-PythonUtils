from MaCh3PythonUtils.file_handling.chain_handler import ChainHandler

from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd
import numpy as np
import pickle
from rich import print

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rich import print

class FileMLInterface(ABC):
    
    def __init__(self, chain: ChainHandler, prediction_variable: str, fit_name: str, scale_features: bool = False, scale_labels: bool = False) -> None:
        """General Interface for all ML models

        :param chain: ChainHandler instance
        :type chain: ChainHandler
        :param prediction_variable: "Label" used for prediction
        :type prediction_variable: str
        :param fit_name: Name of fit, used for output
        :type fit_name: str
        :param scale_features: Whether to scale features, defaults to False
        :type scale_features: bool, optional
        :param scale_labels: Whether to scale labels, defaults to False
        :type scale_labels: bool, optional
        :raises ValueError: Checks to see if label exists in tree
        """        
        self._chain = chain
        
        self._fit_name = fit_name
        self._prediction_variable = prediction_variable
        
        if prediction_variable not in self._chain.ttree_array.columns:
            raise ValueError(f"Cannot find {prediction_variable} in input tree")
        
        self._model = None
        
        # Scaling components        
        self._scaler = StandardScaler(with_mean=scale_features, with_std=scale_features)
        self._label_scaler = StandardScaler(with_mean=scale_labels, with_std=scale_labels)
        
        # Defaults
        self._train_data_indices = np.arange(len(self._chain.ttree_array))
        self._test_data_indices = np.arange(len(self._chain.ttree_array))
            
    def set_training_test_set(self, test_size: float):
        """Splits data/labels into training and testing tests

        :param test_size: Proportion of data used for testing
        :type test_size: float
        """        
        # Splits in training + test_split        
        self._train_data_indices, self._test_data_indices = train_test_split(
            np.arange(len(self._chain.ttree_array)), test_size=test_size, shuffle=True
        )

        # Fit scaling pre-processors on a sample to avoid loading full data into memory
        # Use smaller sample for scaler fitting to reduce memory usage
        sample_size = min(10000, len(self._train_data_indices))
        sample_indices = np.random.choice(self._train_data_indices, size=sample_size, replace=False)
        
        sample_data = self._chain.ttree_array.iloc[sample_indices].drop(columns=[self._prediction_variable])
        sample_labels = self._chain.ttree_array.iloc[sample_indices][[self._prediction_variable]]
        
        self._scaler.fit(sample_data)
        self._label_scaler.fit(sample_labels)
        
        # self._pca_matrix.fit(scaled_training)

    def scale_data(self, input_data):
        # Applies transformations to data set
        scale_data = self._scaler.transform(input_data)
        return scale_data
    
    def scale_labels(self, labels):
        return self._label_scaler.transform(labels)


    def invert_scaling(self, input_data):
        # Inverts transform
        # unscaled_data = self._pca_matrix.inverse_transform(input_data)
        unscaled_data = self._scaler.inverse_transform(input_data)
        return unscaled_data

    @property
    def model(self)->Any:
        """Model used

        :return: Returns ML model being used
        :rtype: Any
        """        
        # Returns model being used
        return self._model    
    
    @property
    def chain(self)->ChainHandler:
        return self._chain

    @property
    def train_labels(self)->pd.Series:
        """Gets training labels

        :return: Training labels set
        :rtype: pd.Series
        """        
        return self.chain.ttree_array.loc[self._train_data_indices, self._prediction_variable]

    @property
    def train_data(self)->pd.DataFrame:
        """Gets training data

        :return: Training data set
        :rtype: pd.DataFrame
        """        
        # Use more memory-efficient indexing with loc for better performance
        data = self.chain.ttree_array.loc[self._train_data_indices]
        return data.drop(columns=[self._prediction_variable])

    @property
    def scaled_train_data(self)->np.ndarray:
        """Gets scaled training data

        :return: Scaled training data set as numpy array to save memory
        :rtype: np.ndarray
        """        
        return self.scale_data(self.train_data)
    
    @property
    def scaled_train_labels(self)->np.ndarray:
        """Gets scaled training labels

        :return: Scaled training labels set as numpy array
        :rtype: np.ndarray
        """        
        labels = np.array(self.train_labels).reshape(-1, 1)
        return self.scale_labels(labels).ravel()
    
    @property
    def test_labels(self)->pd.Series:
        """Gets test labels

        :return: Test labels set
        :rtype: pd.Series
        """        
        return self.chain.ttree_array.loc[self._test_data_indices, self._prediction_variable]


    @property
    def scaled_test_labels(self)->np.ndarray:
        """Gets scaled test labels

        :return: Scaled test labels set as numpy array
        :rtype: np.ndarray
        """        
        labels = np.array(self.test_labels).reshape(-1, 1)
        return self.scale_labels(labels).ravel()


    @property
    def test_data(self)->pd.DataFrame:
        """Gets test data

        :return: Test data set
        :rtype: pd.DataFrame
        """ 
        data = self.chain.ttree_array.loc[self._test_data_indices]
        return data.drop(columns=[self._prediction_variable])

    @property
    def scaled_test_data(self)->np.ndarray:
        """Gets scaled test data

        :return: Scaled test data set as numpy array
        :rtype: np.ndarray
        """        
        return self.scale_data(self.test_data)

    
    def add_model(self, ml_model: Any)->None:
        """Add model to data set

        :param ml_model: Sets model to be ml_model
        :type ml_model: Any
        """        
        # Add ML model into your interface
        self._model = ml_model
    
    @abstractmethod
    def train_model(self):
        """Abstract method, should be overwritten with model training
        """        
        # Train Model method
        pass
    
    @abstractmethod
    def model_predict(self, testing_data: pd.DataFrame):
        """Abstract method, should return model prediction

        :param testing_data: Data to test model on 
        :type testing_data: pd.DataFrame
        """
        pass
        
    def save_model(self, output_file: str):
        """Save model to pickle

        :param output_file: Pickle file to save to
        :type output_file: str
        """
        print(f"Saving to {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(self._model, f)

    def save_scaler(self, output_file: str):
        pickle.dump(self._scaler, open(output_file, 'wb'))
        
    def load_scaler(self, input_scaler: str):
        self._scaler = pickle.load(open(input_scaler, 'rb'))

    def load_model(self, input_model: str):
        """Unpickle model

        :param input_file: Pickled Model
        :type input_file: str
        """        
        print(f"[spring_green1]Attempting to load file from[/spring_green1][bold red3] {input_model}")
        with open(input_model, 'rb') as f:
            self._model = pickle.load(f)
        
    def test_model(self, testing_data: Optional[pd.DataFrame] = None):
        """Test model
        """    
        if self._model is None:
            raise ValueError("No model has been set!")
        
        if testing_data is None:
            testing_data = self.test_data
            
        if testing_data is None:
            raise Exception(f"No test data set!")

        # Convert to numpy array if needed
        if isinstance(testing_data, pd.Series):
            testing_data = testing_data.to_frame().T
        elif hasattr(testing_data, 'values') and np.ndim(testing_data.values) == 1:
            testing_data = pd.DataFrame(testing_data.values.reshape(-1, 1))
            
        return self.model_predict(testing_data)