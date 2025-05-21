from MaCh3PythonUtils.machine_learning.tensorflow.tf_interface import TfInterface
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow.keras as tfk
import keras_tuner as kt

class TfManualInterface(TfInterface):
    _training_settings = {}
            
    def set_training_settings(self, kwargs):
        """Set training settings, needs to be done early for...reasons

        :param kwargs: training kwargs
        :type kwargs: kwargs
        """        
        self._training_settings = kwargs 
        
    def train_model(self):
        """train model
        """ 
        scaled_data = self.scale_data(self._training_data)
        scaled_labels = self.scale_labels(self._training_labels)
        
        lr_schedule = tfk.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.1, min_lr=1e-9, verbose=1)
        stop_early = tfk.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        self._model.fit(scaled_data, scaled_labels, **self._training_settings, callbacks=[lr_schedule, stop_early])


class TfManualLayeredInterface(TfManualInterface):
    __TF_LAYER_IMPLEMENTATIONS = {
        "dense": tfk.layers.Dense,
        "dropout": tfk.layers.Dropout,
        "batchnorm": tfk.layers.BatchNormalization
    }
    
    _layers = []

    def add_layer(self, layer_id: str, layer_args: dict):
        """Add new layer to TF model

        :param layer_id: Layer type [dense/dropout]
        :type layer_id: str
        :param layer_args: kwargs for layer
        :type layer_args: dict
        :raises ValueError: Layer type not implemented in __TF_LAYER_IMPLEMENTATIONS yet
        """        
        if layer_id not in self.__TF_LAYER_IMPLEMENTATIONS.keys():
            raise ValueError(f"{layer_id} not implemented yet!")

        if layer_args.get("kernel_regularizer", False):
            # Hacky, swaps string value of regularliser for proper one
            layer_args["kernel_regularizer"] = tfk.regularizers.L2(layer_args["kernel_regularizer"])


        self._layers.append(self.__TF_LAYER_IMPLEMENTATIONS[layer_id.lower()](**layer_args))
            
