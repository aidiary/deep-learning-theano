from __future__ import print_function

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

from myutils import draw_accuracy, draw_loss

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32

# RGB
img_channels = 3

if __name__ == "__main__":
    # load cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    print('X_train.shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    # build model
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # training
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    if not data_augmentation:
        print('Not using data augmentation.')
        hist = model.fit(X_train, y_train, batch_size=batch_size,
                         nb_epoch=nb_epoch, show_accuracy=True,
                         validation_split=0.1, shuffle=True)
    else:
        pass

    # evaluation
    score = model.evaluate(X_test, y_test,
                           show_accuracy=True, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # draw accuracy/loss plot
    draw_accuracy(hist, title='cifar10_cnn')
    draw_loss(hist, title='cifar10_cnn')
