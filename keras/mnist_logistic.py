from __future__ import print_function
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

batch_size = 600

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
