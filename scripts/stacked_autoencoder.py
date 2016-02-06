#coding: utf-8
import os
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from denoising_autoencoder import DenoisingAutoencoder

class StackedDenoisingAutoencoder(object):
    def __init__(self,
                 numpy_rng,
                 n_ins,
                 hidden_layers_sizes,
                 n_outs,
                 corruption_levels):

        # 隠れ層オブジェクトのリスト
        self.hidden_layers = []

        # 自己符号化器のリスト
        self.autoencoder_layers = []

        # パラメータのリスト
        self.params = []

        # 隠れ層の数
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # 学習データのミニバッチ（入力データと正解ラベル）を表すシンボル
        # これまでの実装と違って複数のメソッド内で使うので属性にしている
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        # ネットワークを構築
        # 隠れ層の数だけループして積み上げていく
        for i in xrange(self.n_layers):
            # ユニット数
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # 隠れ層への入力データ
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.hidden_layers[-1].output

            # 多層パーセプトロンの隠れ層
            # fine-tuningで重みを更新するため
            hidden_layer = HiddenLayer(rng=numpy_rng,
                                       input=layer_input,
                                       n_in=input_size,
                                       n_out=hidden_layers_sizes[i],
                                       activation=T.nnet.sigmoid)
            self.hidden_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)

            # 自己符号化器だが重みは多層パーセプトロンの隠れ層と共有
            # そのため自己符号化器のparamsはない
            # 自己符号化器で重みとバイアスの初期値を求めたあとfine-tuningでそのまま重みとバイアスを引き継げる
            autoencoder_layer = DenoisingAutoencoder(numpy_rng=numpy_rng,
                                                     theano_rng=theano_rng,
                                                     input=layer_input,
                                                     n_visible=input_size,
                                                     n_hidden=hidden_layers_sizes[i],
                                                     W=hidden_layer.W,      # 隠れ層の重みを共有
                                                     bhid=hidden_layer.b)   # 隠れ層のバイアスを共有
            self.autoencoder_layers.append(autoencoder_layer)

        # MNISTの分類ができるように最後にロジスティック回帰層を追加
        self.log_layer = LogisticRegression(
                            input=self.hidden_layers[-1].output,
                            n_in=hidden_layers_sizes[-1],
                            n_out=n_outs)
        self.params.extend(self.log_layer.params)

        # fine-tuning時のコスト関数を計算するシンボル
        # 多層パーセプトロンと同じく負の対数尤度
        self.finetune_cost = self.log_layer.negative_log_likelihood(self.y)

        # 分類の誤差率を計算するシンボル
        self.errors = self.log_layer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        """自己符号化器を学習するpre-training用の関数リストを返す
        教師なし学習なのでxのみを渡す"""
        # 学習に使うミニバッチのインデックス
        index = T.lscalar('index')

        # 複数の自己符号化器で異なる値を指定できるようにシンボル化する
        corruption_level = T.scalar('corruption')
        learning_rate = T.scalar('lr')

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        # 自己符号化器を学習する関数を生成
        # 入力層に近い方から順番に追加する
        pretrain_functions = []
        for autoencoder in self.autoencoder_layers:
            # 誤差と更新式を計算するシンボルを取得
            cost, updates = autoencoder.get_cost_updates(corruption_level, learning_rate)
            fn = theano.function(
                inputs=[
                    index,
                    # Paramにした引数を関数呼び出し時に与えるときはPython変数名ではなく、
                    # Tensorの引数の名前（corruption, lr）で指定できる
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            pretrain_functions.append(fn)

        return pretrain_functions

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        """fine-tuning用の関数を返す"""
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

        index = T.lscalar('index')

        gparams = T.grad(self.finetune_cost, self.params)

        updates = [
            (param, param - gparam * learning_rate) for param, gparam in zip(self.params, gparams)
        ]

        train_model = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            },
            name='train')

        test_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
            },
            name='test')

        valid_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            },
            name='validate')

        def valid_score():
            """各ミニバッチのvalid誤差のリストを返す"""
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def test_score():
            """各ミニバッチのtest誤差のリストを返す"""
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_model, valid_score, test_score

def test_stacked_autoencoder(finetune_lr=0.1, pretraining_epochs=15,
                             pretrain_lr=0.001, training_epochs=200,
                             dataset='mnist.pkl.gz', batch_size=1,
                             hidden_layers_sizes=[1000, 1000, 1000],
                             corruption_levels=[0.1, 0.2, 0.3],
                             valerr_file='validation_error.txt',
                             testerr_file='test_error.txt'):
    datasets = load_data(dataset)
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
    print "getting the pre-training functions ..."
    pretraining_functions = sda.pretraining_functions(train_set_x=train_set_x,
                                                      batch_size=batch_size)

    print "pre-training the model ..."
    start_time = timeit.default_timer()
    for i in xrange(sda.n_layers):
        # pre-trainingのエポック数は固定
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_functions[i](index=batch_index,
                                                  corruption=corruption_levels[i],
                                                  lr=pretrain_lr))
            print "Pre-training layer %i, epoch %d, cost %f" % (i, epoch, np.mean(c))

    end_time = timeit.default_timer()
    training_time = end_time - start_time

    print "The pretraining code for file %s ran for %.2fm" % (os.path.split(__file__)[1], training_time / 60.0)


    # Fine-tuning
    print "getting the fine-tuning functions ..."
    train_model, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print "fine-tuning the model ..."

    # eary-stoppingのパラメータ
    patience = 10 * n_train_batches
    patience_increase = 2.0
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    test_score = 0

    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    fp1 = open(valerr_file, "w")
    fp2 = open(testerr_file, "w")

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses)
                print "epoch %i, minibatch %i/%i, validation error %f %%" % (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100)
                fp1.write("%d\t%f\n" % (epoch, this_validation_loss * 100))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        # 十分改善したならまだ改善の余地があるためpatienceを上げてより多くループを回せるようにする
                        patience = max(patience, iter * patience_increase)
                        print "*** iter %d / patience %d" % (iter, patience)
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = test_model()
                    test_score = np.mean(test_losses)
                    print "    epoch %i, minibatch %i/%i, test error of best model %f %%" % (epoch, minibatch_index + 1, n_train_batches, test_score * 100)
                    fp2.write("%d\t%f\n" % (epoch, test_score * 100))

            # patienceを超えたらループを終了
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    training_time = (end_time - start_time)

    print "Optimization complete with the best validation score of %f %%, on iteration %i, with test performance %f %%" \
            % (best_validation_loss * 100.0, best_iter + 1, test_score * 100)
    print "The training code for file %s ran for %.2fm" % (os.path.split(__file__)[1], training_time / 60.0)

    fp1.close()
    fp2.close()

if __name__ == "__main__":
    test_stacked_autoencoder(dataset="../data/mnist.pkl.gz",
                             hidden_layers_sizes=[1000, 1000, 1000],
                             corruption_levels=[0.1, 0.2, 0.3])
