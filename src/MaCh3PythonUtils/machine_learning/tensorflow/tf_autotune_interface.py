from MaCh3PythonUtils.machine_learning.tensorflow.tf_interface import TfInterface
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow.keras as tfk
import keras_tuner as kt


class TfAutotuneInterface(TfInterface):
    # just make sure things don't break!
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    _epochs = 1000
    _val_split = 0.2

    def build_model(self, **kwargs):
        # Set up layers
        n_layers = kwargs.get("n_layers", [5, 20, 1])

        activation_functions = kwargs.get("layer_activation", ["tanh", "relu"])

        neurons_per_layer = kwargs.get("neurons_per_layer", [16, 2048, 100])
        learning_rate = kwargs.get(
            "learning_rate", [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        )

        regularization_rate = kwargs.get("regularization", [0.001, 0.01, 0.1, 1])

        # Actually make the model
        def model_builder(hp):
            model = tfk.Sequential()

            hp_model_layers = hp.Int(
                "n_layers",
                min_value=n_layers[0],
                max_value=n_layers[1],
                step=n_layers[2],
            )

            for i in range(hp_model_layers):
                # Add layer
                hp_layer_units = hp.Int(
                    f"units_{i}",
                    min_value=neurons_per_layer[0],
                    max_value=neurons_per_layer[1],
                    step=neurons_per_layer[2],
                )
                hp_layer_activation = hp.Choice(
                    f"activation_{i}", values=activation_functions
                )

                hp_layer_regularization = hp.Choice(
                    f"regularization_{i}", regularization_rate
                )

                model.add(
                    tfk.layers.Dense(
                        units=hp_layer_units,
                        activation=hp_layer_activation,
                        kernel_regularizer=tfk.regularizers.L2(hp_layer_regularization),
                    )
                )

                # Add batch normalization
                model.add(tfk.layers.BatchNormalization())

            # Add output
            model.add(tfk.layers.Dense(units=1, activation="linear"))

            hp_learning_rate = hp.Choice("learning_rate", learning_rate)

            model.compile(
                optimizer=tfk.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=kwargs.get("loss", "mse"),
                metrics=kwargs.get("metrics", ["mse", "mae"]),
            )

            return model

        self._epochs = kwargs.get("epochs", 1000)
        self._val_split = kwargs.get("validation_split", 0.2)
        self._batch_size = kwargs.get("batch_size", 2048)

        # Hyperband parameters
        hyperband_iterations = kwargs.get("hyperband_iterations", 100)
        model_directory = kwargs.get("tuning_dir", "tuning-model")
        project_name = kwargs.get("project_name", "tuning-project")

        self._model_tuner = kt.Hyperband(
            model_builder,
            objective="val_loss",
            max_epochs=self._epochs,
            hyperband_iterations=hyperband_iterations,
            directory=model_directory,
            # distribution_strategy=tf.distribute.MirroredStrategy(),
            project_name=project_name,
            overwrite=False,
        )

    def train_model(self):

        scaled_data = self.scale_data(self._training_data)
        scaled_labels = self.scale_labels(self._training_labels)

        stop_early = tfk.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        lr_schedule = tfk.callbacks.ReduceLROnPlateau(
            monitor="loss", patience=5, factor=0.5, min_lr=1e-6, verbose=1
        )

        self._model_tuner.search(
            scaled_data,
            scaled_labels,
            epochs=self._epochs,
            validation_split=self._val_split,
            batch_size=self._batch_size,
            callbacks=[stop_early],
        )

        best_hps = self._model_tuner.get_best_hyperparameters()[0]
        print("Finished auto-tuning")

        self._model = self._model_tuner.hypermodel.build(best_hps)

        self._model.fit(
            scaled_data,
            scaled_labels,
            epochs=self._epochs,
            batch_size=self._batch_size,
            validation_split=0.2,
            callbacks=[stop_early, lr_schedule],
        )
