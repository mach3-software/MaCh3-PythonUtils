from MaCh3PythonUtils.file_handling.chain_handler import ChainHandler

from abc import ABC, abstractmethod
from typing import Any, Tuple, Iterable, Optional
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle

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
        
        self._training_data=None
        self._training_labels=None
        self._test_data=None
        self._test_labels=None

        # Scaling components        
        self._scaler = StandardScaler(with_mean=scale_features, with_std=scale_features)
        self._label_scaler = StandardScaler(with_mean=scale_labels, with_std=scale_labels)
        
            
    def __separate_dataframe(self)->Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data frame into feature + label objects

        :return: features, labels
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """        
        # Separates dataframe into features + labels
        features = self._chain.ttree_array.copy()
        labels   = pd.DataFrame(features.pop(self._prediction_variable) )
        
        return features, labels
    
    def set_training_test_set(self, test_size: float):
        """Splits data/labels into training and testing tests

        :param test_size: Proportion of data used for testing
        :type test_size: float
        """        
        # Splits in traing + test_spit
        features, labels = self.__separate_dataframe()
        self._training_data, self._test_data, self._training_labels, self._test_labels =  train_test_split(features, labels, test_size=test_size)

        # Fit scaling pre-processors. These get applied properly when scale_data is called
        self._scaler.fit(self._training_data)
        self._label_scaler.fit(self._training_labels)
        
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
    def training_labels(self)->pd.DataFrame:
        """Gets training labels

        :return: Training labels set
        :rtype: pd.DataFrame
        """        
        if self._training_labels is None:
            return self._chain.ttree_array.iloc[:,-1]

        return self._training_labels

    @property
    def training_data(self)->pd.DataFrame:
        """Gets training data

        :return: Training data set
        :rtype: pd.DataFrame
        """        
        if self._training_data is None:
            return self._chain.ttree_array.iloc[:,:-1]

        return self._training_data

    @property
    def scaled_training_data(self)->pd.DataFrame:
        """Gets scaled training data

        :return: Scaled training data set
        :rtype: pd.DataFrame
        """        
        if self._training_data is None:
            return self._chain.ttree_array.iloc[:,:-1]

        return self.scale_data(self._training_data)
    
    @property
    def scaled_training_labels(self)->pd.DataFrame:
        """Gets scaled training labels

        :return: Scaled training labels set
        :rtype: pd.DataFrame
        """        
        if self._training_labels is None:
            return self._chain.ttree_array.iloc[:,-1]

        return self.scale_labels(self._training_labels)
    
    @property
    def test_labels(self)->pd.DataFrame:
        """Gets test labels

        :return: Test labels set
        :rtype: pd.DataFrame
        """        
        if self._test_labels is None:
            return self._chain.ttree_array.iloc[:,-1]
        return self._test_labels

    @property
    def scaled_test_labels(self)->pd.DataFrame:
        """Gets scaled test labels

        :return: Scaled test labels set
        :rtype: pd.DataFrame
        """        
        if self._test_labels is None:
            return self._chain.ttree_array.iloc[:,-1]
        
        return self.scale_labels(self._test_labels)


    @property
    def test_data(self)->pd.DataFrame:
        """Gets training data

        :return: Training data set
        :rtype: pd.DataFrame
        """ 
        if self._test_data is None:
            return self._chain.ttree_array.iloc[:,:-1]
        
        return self._test_data

    @property
    def scaled_test_data(self)->pd.DataFrame:
        """Gets scaled test data

        :return: Scaled test data set
        :rtype: pd.DataFrame
        """        
        if self._test_data is None:
            return self._chain.ttree_array.iloc[:,:-1]
        return self.scale_data(self._test_data)

    
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
    def model_predict(self, testing_data: pd.DataFrame)->Iterable:
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
        
    def test_model(self, testing_data: Optional[pd.DataFrame] = None)->Iterable:
        """Test model
        """    
        if self._model is None:
            raise ValueError("No model has been set!")
        
        if testing_data is None and self._test_data is not None:
            testing_data = self._test_data
        else:
            raise Exception(f"[bold red3]No test data set![/bold red3]")            

        if np.ndim(testing_data) == 1:
            testing_data = testing_data.reshape(-1, 1)
            
        return self.model_predict(testing_data)