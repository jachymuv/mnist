import sys
import random

import numpy as np

# import tensorflow as tf
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

# print(tf.version)


class MnistProc:
    def __init__(
        self,
        *,
        path=None,
        limit=None,
        early_stopping=True,
        lr_scheduler=False,
        verbose=False,
        save_model=None,
        max_epochs=50,
        batch_size=128,
        seed=None,
    ):

        self.verbose = verbose
        self.early_stopping = early_stopping
        self.lr_scheduler = lr_scheduler

        if self.verbose:
            print("MnistProc with TF version", TFVERSION, file=sys.stderr)
        else:
            set_verbosity(TFERROR)

        self.data = MnistData(path=path, limit=limit)
        self.model = Sequential()
        self.compiled = False
        self.fitted = False
        self.callbacks = []
        self.save_model = save_model
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        if seed:
            self.set_random_seeds(seed)

    def compile_model(self, name="cnn9975"):

        self._add_callbacks()

        if name == "cnn9975":
            self._compile_cnn9975_model()
            self.compiled = True
        else:
            print("ERROR: Model name %s unknown, model not compiled", file=sys.stderr)

    def _compile_cnn9975_model(self):

        self.model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), activation="relu"))
        self.model.add(BatchNormalization())
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

        # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        if self.verbose:
            self.model.summary()

    def _add_callbacks(self):

        if self.early_stopping:
            es = EarlyStopping(monitor="val_acc", patience=5)
            self.callbacks.append(es)

        if self.lr_scheduler:
            lrs = LearningRateScheduler(lambda x: 0.001 * 0.95 ** x)
            self.callbacks.append(lrs)

    def train_model(self, val_split=0.2):

        if self.compiled:
            self.history = self.model.fit(
                self.data.X_train,
                self.data.y_train_cat,
                batch_size=self.batch_size,
                epochs=self.max_epochs,
                verbose=self.verbose,
                validation_split=val_split,
                callbacks=self.callbacks,
            )
            self.fitted = True

            if self.save_model:
                self.model.save(self.save_model)
        else:
            print("ERROR: Model not compiled and cannot be trained", file=sys.stderr)

    def get_error_rate(self):

        if self.fitted:
            return 1.0 - self.model.evaluate(self.data.X_test, self.data.y_test_cat, verbose=self.verbose)[1]
        else:
            print("ERROR: Model not fitted and cannot be evaluated", file=sys.stderr)
            return None

    def load_model(self, path):

        self.model = load_model(path)
        self.compiled = True
        self.fitted = True

    def do_all(self):

        self.compile_model()
        self.train_model()
        return self.get_error_rate()

    def set_random_seeds(self, seed):

        np.random.seed(seed)
        random.seed(seed)
        set_random_seed(seed)
