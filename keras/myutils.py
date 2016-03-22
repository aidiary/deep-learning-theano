from __future__ import print_function
import matplotlib.pyplot as plt

def draw_accuracy(hist, title=None, filename=None):
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']

    nb_epoch = len(acc)

    plt.plot(range(nb_epoch), acc, marker='.', label='acc')
    plt.plot(range(nb_epoch), val_acc, marker='.', label='val_acc')
    plt.legend(loc='best', fontsize=10)
    plt.grid()

    if title:
        plt.title(title)

        plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.tight_layout()

    if filename:
        plt.savefig(filename)

    plt.show()

def draw_loss(hist, title=None, filename=None):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    nb_epoch = len(loss)

    plt.plot(range(nb_epoch), loss, marker='.', label='loss')
    plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best', fontsize=10)
    plt.grid()

    if title:
        plt.title(title)

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()

    if filename:
        plt.savefig(filename)

    plt.show()
