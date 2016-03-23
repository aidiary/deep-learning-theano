from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, AutoEncoder
from keras.optimizers import RMSprop
from keras.utils import np_utils

nb_classes = 10

if __name__ == '__main__':
    # load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784).astype('float32')
    X_test = X_test.reshape(10000, 784).astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    # convert class vectors to 1-of-K format
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    print('train samples: ', X_train.shape)
    print('test samples: ', X_test.shape)

    # building the model
    encoder = Sequential([Dense(500, input_dim=784)])
    decoder = Sequential([Dense(784, input_dim=500)])
    autoencoder = AutoEncoder(encoder=encoder, decoder=decoder,
                              output_reconstruction=True)
    model = Sequential()
    model.add(autoencoder)

    # training the autoencoder
    model.compile(optimizer='sgd', loss='mse')
    model.fit(X_train, X_train, nb_epoch=10)

    # output hidden layer
    autoencoder.output_reconstruction = False
    model.compile(optimizer='sgd', loss='mse')
    representations = model.predict(X_test)

    print(representations.shape)

