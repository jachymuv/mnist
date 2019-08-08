from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle


class MnistData:
    def __init__(self, *, path=None, limit=None):

        if path:
            (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data(path)
        else:
            (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        print(y_train[:20])

        # shuffle data order
        self.shuffle_train()

        if limit:
            self._limit_train(limit)
        self._reshape_normalize()

    def shuffle_train(self):
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)

    def _limit_train(self, limit):

        if limit:
            self.X_train = self.X_train[:limit]
            self.y_train = self.y_train[:limit]

    def _reshape_normalize(self):

        self.X_train = self.X_train.astype("float32") / 255.0
        self.X_test = self.X_test.astype("float32") / 255.0

        self.X_train = self.X_train.reshape(-1, 28, 28, 1)
        self.X_test = self.X_test.reshape(-1, 28, 28, 1)

        self.y_train_cat = to_categorical(self.y_train, 10)
        self.y_test_cat = to_categorical(self.y_test, 10)
