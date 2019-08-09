"""
Core file of MnistProc package implementing the MnistProc class
"""

import sys
import random

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.python.framework.versions import VERSION as TFVERSION
from tensorflow.keras.models import load_model
from tensorflow import set_random_seed
from tensorflow.logging import set_verbosity
from tensorflow.logging import ERROR as TFERROR

from .data import MnistData


class MnistProc:
    """
    Main class of the MnistProc package
    implementing building, compiling, training, and testing of the model
    """

    def __init__(
        self,
        *,  # force use of keyword parameter names
        data_path: str = None,  # path where downloaded MNIST data should be stored
        limit: int = None,  # max size of training data
        early_stopping: bool = True,  # use early stopping callback
        lr_scheduler: bool = False,  # use learning rate scheduling
        verbose: bool = False,  # verbose or not
        save_model: str = None,  # path where to save the model (or do not save)
        max_epochs: int = 50,  # maximum number of training epochs
        batch_size: int = 128,  # batch size
        val_split: float = 0.1,  # portion of data used for validation during training
        seed: int = None,  # random number generator seed
    ):

        self.verbose = verbose
        self.early_stopping = early_stopping
        self.lr_scheduler = lr_scheduler
        self.data = MnistData(path=data_path, limit=limit)  # build Mnist datasets
        self.model = Sequential()  # init model as Keras Sequential
        self.compiled = False
        self.fitted = False
        self.callbacks = []
        self.save_model = save_model
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.val_split = val_split

        if self.verbose:
            print("MnistProc with TF version", TFVERSION, file=sys.stderr)
        else:
            set_verbosity(TFERROR)

        if seed:
            self.set_random_seeds(seed)

    def compile_model(self, name="cnn9975"):
        """
        Compile model specified by 'name'
        currently only "cnn9975" implemented
        """

        self._add_callbacks()

        if name == "cnn9975":
            self._compile_cnn9975_model()
            self.compiled = True
        else:
            print("ERROR: Model name %s unknown, model not compiled", file=sys.stderr)

    def _compile_cnn9975_model(self):
        """
        Builds and compile MNIST digit recognition model in TensorFlow
        It is inspired by cnn9975 (model achieving 99.75 acc in Kaggle competition),
        but the model is kept lightweight and all features are implemented

        Features implemented:
        nonlinear convolution layers, 
        learnable pooling layers,
        ReLU activation, 
        dropout,
        batch normalization,
        adam optimization
        """

        # Convolutional layers with batch normalization
        self.model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), activation="relu"))
        self.model.add(BatchNormalization())

        # instead of MaxPooling layer, use Conv filters of size 5 and stride 2 and padding
        # to implement learnable pooling
        self.model.add(Conv2D(32, (5, 5), strides=2, padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(64, (3, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3, 3), activation="relu"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, (5, 5), strides=2, padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(128, (4, 4), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dropout(0.4))
        self.model.add(Dense(10, activation="softmax"))

        # compile using adam optimization and categorical cross entropy loss
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        if self.verbose:
            self.model.summary()

    def _add_callbacks(self):
        """
        Add callbacks (early stopping and learning rate scheduling)
         if specified by instance attributes
        """

        if self.early_stopping:
            es = EarlyStopping(monitor="val_acc", patience=5)
            self.callbacks.append(es)

        if self.lr_scheduler:
            lrs = LearningRateScheduler(lambda x: 0.001 * 0.95 ** x)
            self.callbacks.append(lrs)

    def train_model(self):
        """
        Train model as specified by instance attributes
        """

        # check whether model was compiled
        if self.compiled:
            self.history = self.model.fit(
                self.data.X_train,
                self.data.y_train_cat,
                batch_size=self.batch_size,
                epochs=self.max_epochs,
                verbose=self.verbose,
                validation_split=self.val_split,
                callbacks=self.callbacks,
            )
            self.fitted = True

            if self.save_model:
                self.model.save(self.save_model)
        else:
            print("ERROR: Model not compiled and cannot be trained", file=sys.stderr)

    def get_error_rate(self) -> float:
        """
         Run model evaluation and return error rate
        """

        # Check whether model was already fitted
        if self.fitted:
            return (
                1.0
                - self.model.evaluate(self.data.X_test, self.data.y_test_cat, verbose=self.verbose)[
                    1
                ]
            )
        else:
            print("ERROR: Model not fitted and cannot be evaluated", file=sys.stderr)
            return None

    def load_model(self, path):
        """
        Load model specified by path
        """

        self.model = load_model(path)
        self.compiled = True
        self.fitted = True

    def do_all(self) -> float:
        """
        Convenience function performing model compilation, training and evaluation
        Returs error rate value
        """

        self.compile_model()
        self.train_model()

        return self.get_error_rate()

    def set_random_seeds(self, seed: int):
        """
        To support experiment reproducibility, 
        set random seed of Python, Numpy and Tensorflow to a specified value

        Reproducibility however cannot be guaranteed when multithreading is used

        Params: 
         - seed (int) - number to used as a seed for all 3 values 
        """

        random.seed(seed)
        np.random.seed(seed)
        set_random_seed(seed)
