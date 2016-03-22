from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from myutils import draw_accuracy, draw_loss

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28

# number of convolutional filters to use
nb_filters = 32

# size of pooling area for max pooling
nb_pool = 2

# convolution kernel size
nb_conv = 3

if __name__ == '__main__':
    # load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols).astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    # convert class vectors to 1-of-K format
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    print('train samples: ', X_train.shape)
    print('test samples: ', X_test.shape)

    # building the model
    print('building the model ...')

    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    # training
    hist = model.fit(X_train, y_train,
                     batch_size=batch_size,
                     nb_epoch=nb_epoch,
                     show_accuracy=True,
                     verbose=1,
                     validation_split=0.1)

    # evaluation
    score = model.evaluate(X_test, y_test,
                           show_accuracy=True, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # draw accuracy/loss plot
    draw_accuracy(hist, title='cnn')
    draw_loss(hist, title='cnn')
