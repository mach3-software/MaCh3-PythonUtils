# Let's make a tensor flow interface!
from typing import Any
from machine_learning.fml_interface import FmlInterface
import tensorflow as tf
from file_handling.chain_handler import ChainHandler

class TfInterface(FmlInterface):
    __TF_LAYER_IMPLEMENTATIONS = {
        "dense": tf.keras.layers.Dense,
        "dropout": tf.keras.layers.Dropout
    }
    
    def __init__(self, chain: ChainHandler, prediction_variable: str) -> None:
        super().__init__(chain, prediction_variable)
        
        self._model = None
        self._layers = []
        self._training_settings = {}
        
        
    def add_layer(self, layer_id, layer_args):
        if layer_id not in self.__TF_LAYER_IMPLEMENTATIONS.keys():
            raise ValueError(f"{layer_id} not implemented yet!")

        self._layers.append(self.__TF_LAYER_IMPLEMENTATIONS[layer_id.lower()](**layer_args))
    
    def build_model(self, kwargs_dict):
        if self._model is None or not self._layers:
            raise ValueError("No model can be built! Please setup model and layers")
        
        for layer in self._layers:
            self._model.add(layer)
            
        self._model.build()
        
        self._model.compile(**kwargs_dict)
            
    def set_training_settings(self, kwargs):
        self._training_settings = kwargs 
        
    def train_model(self):
        self._model.fit(self._training_data, self._training_labels, **self._training_settings)
    
    def save_model(self, output_file: str):
        self._model.export(output_file)
        
    def load_model(self, input_file: str):
        self._model = tf.saved_model.load(input_file)
    
    def model_predict(self, testing_data):
        return self._model.predict_on_batch(testing_data).reshape(1,-1)
