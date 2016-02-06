#coding: utf-8
import numpy as np

from logistic_sgd import load_data
from stacked_autoencoder import StackedDenoisingAutoencoder

def test_stacked_autoencoder(finetune_lr=0.1, pretraining_epochs=15,
                             pretrain_lr=0.001, training_epochs=100,
                             dataset='mnist.pkl.gz', batch_size=1,
                             hidden_layers_sizes=[1000, 1000, 1000],
                             corruption_levels=[0.1, 0.2, 0.3],
                             pretrain_flag=True,
                             testerr_file='test_error.txt'):
    datasets = load_data("../data/mnist.pkl.gz")
    train_set_x = datasets[0][0]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    numpy_rng = np.random.RandomState(89677)

    print "building the model ..."

    sda = StackedDenoisingAutoencoder(
        numpy_rng,
        28 * 28,
        hidden_layers_sizes,
        10,
        corruption_levels)

    # Pre-training
    if pretrain_flag:
        print "getting the pre-training functions ..."
        pretraining_functions = sda.pretraining_functions(train_set_x=train_set_x,
                                                      batch_size=batch_size)

        print "pre-training the model ..."
        for i in xrange(sda.n_layers):
            for epoch in xrange(pretraining_epochs):
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_functions[i](index=batch_index,
                                                      corruption=corruption_levels[i],
                                                      lr=pretrain_lr))
                print "Pre-training layer %i, epoch %d, cost %f" % (i, epoch, np.mean(c))

    # Fine-tuning
    print "getting the fine-tuning functions ..."
    train_model, _, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print "fine-tuning the model ..."

    epoch = 0
    fp = open(testerr_file, "w")
    while (epoch < training_epochs):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            train_model(minibatch_index)
        test_losses = test_model()
        test_score = np.mean(test_losses)
        print "Fine-tuning, epoch %d, test error %f" % (epoch, test_score * 100)
        fp.write("%d\t%f\n" % (epoch, test_score * 100))
    fp.close()

def test_denoising():
    test_stacked_autoencoder(hidden_layers_sizes=[1000, 1000, 1000],
                             corruption_levels=[0, 0, 0],
                             testerr_file="sa_error_without_denoising.txt")
    test_stacked_autoencoder(hidden_layers_sizes=[1000, 1000, 1000],
                             corruption_levels=[0.1, 0.2, 0.3],
                             testerr_file="sa_error_with_denoising.txt")

def test_pretraining():
    test_stacked_autoencoder(hidden_layers_sizes=[1000, 1000, 1000],
                             corruption_levels=[0.1, 0.2, 0.3],
                             testerr_file="sa_error_with_pretraining.txt",
                             pretrain_flag=True)
    test_stacked_autoencoder(hidden_layers_sizes=[1000, 1000, 1000],
                             corruption_levels=[0.1, 0.2, 0.3],
                             testerr_file="sa_error_without_pretraining.txt",
                             pretrain_flag=False)

if __name__ == "__main__":
    # 雑音除去自己符号化器の効果を検証
    test_denoising()

    # 事前学習の効果を検証
    test_pretraining()
