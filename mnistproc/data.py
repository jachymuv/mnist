"""
This file implements class MnistData
which takes care of downloading and preprocessing MNIST handwritten digit dataset
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle


class MnistData:
    """
    Preprocessing of MNIST handwritten digit dataset
    """

    def __init__(
        self,
        *,  # force use of keyword parameter names
        path: str = None,  # path to where the MNIST dataset should be stored
        limit: int = None,  # limit on the maximum size of the training set
    ):

        if path:
            (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data(path)
        else:
            (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        if limit:
            self._limit_train(limit)
        self._reshape_normalize()

    def shuffle_train(self):
        """
        Shuffle the order of the training set
        As the MNIST dataset version downloaded by this class seems to be already shuffled,
        it is not called by default
        """

        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)

    def _limit_train(self, limit: int):
        """
        Limit the size of the training set
        Params: limit(int) - maximum size of the training set
        """

        if limit:
            self.X_train = self.X_train[:limit]
            self.y_train = self.y_train[:limit]

    def _reshape_normalize(self):
        """
        Normalize values in the MNIST set,
        reshape to 28x28 shape,
        create 1-HoT target labels
        """

        self.X_train = self.X_train.astype("float32") / 255.0
        self.X_test = self.X_test.astype("float32") / 255.0

        self.X_train = self.X_train.reshape(-1, 28, 28, 1)
        self.X_test = self.X_test.reshape(-1, 28, 28, 1)

        self.y_train_cat = to_categorical(self.y_train, 10)
        self.y_test_cat = to_categorical(self.y_test, 10)
