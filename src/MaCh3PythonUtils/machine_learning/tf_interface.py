# Let's make a tensor flow interface!
from MaCh3PythonUtils.machine_learning.file_ml_interface import FileMLInterface
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow.keras as tfk

class TfInterface(FileMLInterface):
    __TF_LAYER_IMPLEMENTATIONS = {
        "dense": tfk.layers.Dense,
        "dropout": tfk.layers.Dropout,
        "batchnorm": tfk.layers.BatchNormalization
    }
    
    __TF_REGULARIZERS = {
        "l2" : tf.keras.regularizers.L2
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

        if "kernel_regularizer" in layer_args.keys():
            # Hacky, swaps string value of regularliser for proper one
            reg = layer_args["kernel_regularizer"]
            reg_name = list(reg.keys())[0]
            layer_args["kernel_regularizer"] = self.__TF_REGULARIZERS[reg_name.lower()](reg[reg_name])

        self._layers.append(self.__TF_LAYER_IMPLEMENTATIONS[layer_id.lower()](**layer_args))
            
    def build_model(self, model_args: dict):
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
        optimizer = tfk.optimizers.AdamW(learning_rate=model_args.get("learning_rate", 1e-3), 
                          weight_decay=1e-4, clipnorm=1.0)

        self._model.compile(**model_args, optimizer=optimizer)
            
    
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
    
    def save_model(self, output_file: str):
        """Save model to file

    :param output_file: Output file to save model to
        :type output_file: str
        """        
        if not ".keras" in output_file:
            output_file+=".keras"
        
        self._model.save(output_file,)
        
    def load_model(self, input_file: str):
        """Load model from file

        :param input_file: Name offile to load model from
        :type input_file: str
        """        
        
        print(f"Loading model from {input_file}")
        self._model = tf.keras.models.load_model(input_file)
    
    def model_predict(self, test_data: pd.DataFrame):
        """Get model prediction

        :param test_data: data to run model over for prediction
        :type test_data: pd.DataFrame
        :return: model predction
        :rtype: NDArray
        """        
        # Hacky but means it's consistent with sci-kit interface
        scaled_data = self.scale_data(test_data)

        if self._model is None:
            return np.zeros(len(test_data))
        
        return self._model.predict(scaled_data, verbose=False).T[0]


    def model_predict_no_scale(self, test_data):
        # Same as above but specifically for TF, optimised to avoid if statement...
        return self._model(test_data, training=False)
    