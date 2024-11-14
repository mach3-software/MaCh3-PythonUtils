from MaCh3PythonUtils.machine_learning.tf_interface import TfInterface
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow.keras as tfk
import keras_tuner as kt


class TfManualInterface(TfInterface):
    __TF_LAYER_IMPLEMENTATIONS = {
        "dense": tfk.layers.Dense,
        "dropout": tfk.layers.Dropout,
        "batchnorm": tfk.layers.BatchNormalization
    }
    
    _layers = []
    _training_settings = {}
    
    
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

        if layer_args.get("kernel_regularizer"):
            # Hacky, swaps string value of regularliser for proper one
            layer_args["kernel_regularizer"] = tfk.regularizers.L2(0.2)

        self._layers.append(self.__TF_LAYER_IMPLEMENTATIONS[layer_id.lower()](**layer_args))
            
    def build_model_sequential(self, model_args: dict):
        """Build and compile TF model

        :param model_args: Model arguments as dictionary
        :type model_args: dict
        :raises ValueError: Model not set up yet
        """

        if self._model is None or not self._layers:
            raise ValueError("No model can be built! Please setup model and layers")
        
        for layer in self._layers:
            self._model.add(layer)
            
        self._model.build()
        optimizer = tfk.optimizers.AdamW(learning_rate=model_args.get("learning_rate", 1e-5),
                        weight_decay=1e-4, clipnorm=1.0)
        
        _ = model_args.pop("learning_rate", None)

        
        self._model.compile(**model_args, optimizer=optimizer)

    def build_model_residual(self, model_args: dict):
        input_shape = self.training_data.shape[1:]  # Assuming shape is (batch_size, features)
        network_input = tfk.layers.Input(shape=input_shape)
        
        # Initial layer
        x = self._layers[0](network_input)
        
        # Apply each layer in self._layers, adding residual connections every other layer
        for i, layer in enumerate(self._layers[1:], start=1):
            # Apply layer
            x_new = layer(x)
            
            # Add residual connection every other layer if shapes match
            if i % 2 == 0 and x.shape == x_new.shape:
                x = tfk.layers.add([x, x_new])
            else:
                x = x_new  # Update x with current layer output
        
        # Define and compile the model
        self._model = tfk.Model(inputs=network_input, outputs=x)
        optimizer = tfk.optimizers.AdamW(learning_rate=model_args.get("learning_rate", 1e-5),
                        weight_decay=1e-4, clipnorm=1.0)
        
        _ = model_args.pop("learning_rate", None)
        
        self._model.compile(optimizer=optimizer, **model_args)

        
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
        
        lr_schedule = tfk.callbacks.ReduceLROnPlateau(monitor="loss", patience=5, factor=0.5, min_lr=1e-6, verbose=1)

        self._model.fit(scaled_data, self._training_labels, **self._training_settings, callbacks=[lr_schedule])
        print(f"Using loss function: {self._model.loss}")  
