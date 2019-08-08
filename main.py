#!/usr/bin/env python3

import argparse

from mnistproc.core import MnistProc

# from mnistproc.data import MnistData

if __name__ == "__main__":
    mp = MnistProc(limit=500, verbose=True, save_model="cnn9975.h5", max_epochs=2, seed=1)
    # mp.compile_model()
    # mp.train_model()
    mp.load_model("cnn9975.h5")
    acc = mp.get_accuracy()

    print("Test accuracy", 100.0 * acc)
    # print(mp.data.y_train_cat)
    # data = MnistData()
