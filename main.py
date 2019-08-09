#!/usr/bin/env python3
"""
Main file providing command line interface to test the MnistProc package
There are 3 test scenarios: {all, small, test} corresponding to:
- training/testing on all MNIST data
- training on smaller data (size can be specified), test on all the test set
- load and test a pretrained model

Run pyhon3 main.py -h to get help on command line parameters
"""

import argparse
import os.path
from enum import Enum

from mnistproc.core import MnistProc


class RunType(Enum):
    """
    Helper class for argparse to enumerate test scenarios
    """

    all = "all"
    small = "small"
    test = "test"

    def __str__(self):
        return self.value


if __name__ == "__main__":

    # Parse command line arguments
    arg_parser = argparse.ArgumentParser(
        description="""Train and test CNN9975-like model for MNIST handwriten digit recognition"""
    )

    arg_parser.add_argument(
        "run_type", type=RunType, choices=list(RunType), help="type of run (all, small, test)"
    )

    arg_parser.add_argument(
        "-l",
        "--load_model",
        action="store",
        help="""path to the hdf5 file from which an already trained model will be loaded""",
    )

    arg_parser.add_argument(
        "-s",
        "--save_model",
        action="store",
        help="""after training, save the model to the hdf5 file specified by this path""",
    )

    arg_parser.add_argument(
        "-n",
        "--train_limit",
        action="store",
        type=int,
        help="""limit training data size (runtype 'small' required)""",
    )

    arg_parser.add_argument(
        "-e", "--max_epochs", action="store", type=int, default=50, help="""max number of epochs"""
    )

    arg_parser.add_argument("-d", "--seed", action="store", type=int, help="""set random seed""")

    arg_parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    params = arg_parser.parse_args()

    # Limited training set size is only allowed when 'small'
    if params.train_limit and params.run_type != RunType.small:
        arg_parser.error("--train_limit requires run_type small")

    # Default train size for small
    if params.run_type == RunType.small and not params.train_limit:
        params.train_limit = 1000

    # Check whether the model file exists
    if params.load_model:
        if not os.path.isfile(params.load_model):
            arg_parser.error("model file %s does not exist" % params.load_model)

    if params.verbose:
        verbosity = 1
    else:
        verbosity = 0

    # fork based on run_type {all, small, test}
    if params.run_type == RunType.all:
        mp = MnistProc(
            verbose=verbosity,
            save_model=params.save_model,
            max_epochs=params.max_epochs,
            seed=params.seed,
        )
        error_rate = mp.do_all()

    elif params.run_type == RunType.small:
        mp = MnistProc(
            limit=params.train_limit,
            verbose=verbosity,
            save_model=params.save_model,
            max_epochs=params.max_epochs,
            seed=params.seed,
        )
        error_rate = mp.do_all()

    elif params.run_type == RunType.test:
        mp = MnistProc(verbose=verbosity)
        mp.load_model(params.load_model)
        error_rate = mp.get_error_rate()

    # Print error rate on the test set
    print("Test error rate: %.3f %%" % (100.0 * error_rate))
