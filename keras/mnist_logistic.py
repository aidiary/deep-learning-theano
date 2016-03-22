from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

from myutils import draw_accuracy, draw_loss

batch_size = 600
nb_in = 784
nb_out = 10
nb_epoch = 100

if __name__ == "__main__":
    # load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, nb_in).astype('float32')
    X_test = X_test.reshape(10000, nb_in).astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    # convert class vectors to 1-of-K format
    y_train = np_utils.to_categorical(y_train, nb_out)
    y_test = np_utils.to_categorical(y_test, nb_out)

    print('train samples: ', X_train.shape)
    print('test samples: ', X_test.shape)

    # building the model
    print('building the model ...')

    model = Sequential()
    model.add(Dense(10, input_shape=(784,)))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.13)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    # training
    hist = model.fit(X_train, y_train,
                     batch_size=batch_size,
                     nb_epoch=nb_epoch,
                     show_accuracy=True,
                     verbose=1,
                     validation_split=0.1)

    # evaluate
    score = model.evaluate(X_test, y_test,
                           show_accuracy=True, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # draw accuracy/loss plot
    draw_accuracy(hist, title='logistic regression')
    draw_loss(hist, title='logistic regression')
