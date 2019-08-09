# MNIST Handwritten Digit Recognition 

This repository contains code to train and test an ML model for the MNIST handwritten digit recognition task. The model implemented is a lightweighted version of the Kaggle-leading model [CNN9975](https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist). The CNN9975 model build upon the idea of the well-known LeCun et al. paper _"Gradient-Based Learning Applied to Document Recognition"_ from 1998, applying several methods that prove to improve the performance of CNN-based systems. The authors report 99.75% accuracy which is very close to the currently best reported result of 99.80%. As this program is only an exercise not meant to run on a powerful hawrdware, it uses only a subset of these improvements, namely:

* nonlinear convolution layers 
* learnable pooling layers
* ReLU activation
* dropout
* batch normalization
* adam optimization

Early stopping and decaying learning rate are implemented as optional features. Other advanced techniques used in CNN9975, data augmentation and ensembling/bagging, were for the moment left out but can be added in the future. The implementation is based on TensorFlow/Keras and was tested with TensorFlow versions 1.14.0 and 2.0.0 Alpha0 using Python 3.7. The accuracy of the implemented model is in the range of 99.35--99.55%, depending on the random factors during training. 

## Repository Content:

* mnistproc/ - main package implementing all modeling and preprocessing of MNIST data 
* models/ - examples of trained models 
* main.py - the main scripts providing a command line interface for MnistProc
* requirements.txt - requirements file when using a GPU
* requirements-cpu.txt - requirements file when using a GPU
* README.MD - this file

## Install

It is recommended to create a virtual environment to run the program. 

_Example_:

<pre>
`python3 -m venv env`
`source env/bin/activate` (on Windows `env\Scripts\activate`)
`pip install -r requirements.txt` (or `requirements-cpu.txt` for CPU version) 
</pre>

## Usage

<pre>
usage: main.py [-h] [-l LOAD_MODEL] [-s SAVE_MODEL] [-n TRAIN_LIMIT]
               [-e MAX_EPOCHS] [-x] [-d SEED] [-v]
               {all,small,test}

positional arguments:
  {all,small,test}      type of run (all, small, test)

optional arguments:
  -h, --help            show this help message and exit
  -l LOAD_MODEL, --load_model LOAD_MODEL
                        path to the hdf5 file from which an already trained
                        model will be loaded
  -s SAVE_MODEL, --save_model SAVE_MODEL
                        after training, save the model to the hdf5 file
                        specified by this path
  -n TRAIN_LIMIT, --train_limit TRAIN_LIMIT
                        limit training data size (runtype 'small' required)
  -e MAX_EPOCHS, --max_epochs MAX_EPOCHS
                        max number of epochs
  -x, --early_stopping  turn on early stopping with patience
  -d SEED, --seed SEED  set random seed
  -v, --verbose         verbose output
</pre>

So the main point is there are prepared three run scenarios in _main.py_:

* _all_ - training/testing on all MNIST data
* _small_ - training on smaller data (size can be specified), test on all the test set
* _test_ - load and test a pretrained model

Other parameters, like maximum number of epochs, early stopping, training size data limit 

### Example Calls

**all** :

 `./main.py all --seed 1 -s models/model1.h5`
 
 _train/test on all data with random seed 1 and then save the model to a file_

**small** :

 `./main.py small -v -n 300 -e 2`
 
 _run in verbose, limit training data to 300 samples and train only for two epochs, then test the model (i.e. sanity check)_

**test** :

 `./main.py test -l models/model_es.h5`
 
 _test the model in the specified file_
 
