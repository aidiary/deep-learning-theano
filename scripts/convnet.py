#coding: utf-8
import os
import time
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

class LeNetConvPoolLayer(object):
    """畳み込みニューラルネットの畳み込み層＋プーリング層"""
    def __init__(self, rng, input, image_shape, filter_shape, poolsize=(2, 2)):
        # 入力の特徴マップ数は一致する必要がある
        assert image_shape[1] == filter_shape[1]

        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize)

        W_bound = np.sqrt(6.0 / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                       dtype=theano.config.floatX),  # @UndefinedVariable
            borrow=True)

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)  # @UndefinedVariable
        self.b = theano.shared(value=b_values, borrow=T)

        # 入力の特徴マップとフィルタの畳み込み
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape)

        # Max-poolingを用いて各特徴マップをダウンサンプリング
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True)

        # バイアスを加える
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='mnist.pkl.gz', batch_size=500):
    rng = np.random.RandomState(23455)

    # 学習データのロード
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # ミニバッチの数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print "building the model ..."

    # ミニバッチのインデックスを表すシンボル
    index = T.lscalar()

    # ミニバッチの学習データとラベルを表すシンボル
    x = T.matrix('x')
    y = T.ivector('y')

    # 入力
    # 入力のサイズを4Dテンソルに変換
    # batch_sizeは訓練画像の枚数
    # チャンネル数は1
    # (28, 28)はMNISTの画像サイズ
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # 最初の畳み込み層+プーリング層
    # 畳み込みに使用するフィルタサイズは5x5ピクセル
    # 畳み込みによって画像サイズは28x28ピクセルから24x24ピクセルに落ちる
    # プーリングによって画像サイズはさらに12x12ピクセルに落ちる
    # 特徴マップ数は20枚でそれぞれの特徴マップのサイズは12x12ピクセル
    # 最終的にこの層の出力のサイズは (batch_size, 20, 12, 12) になる
    layer0 = LeNetConvPoolLayer(rng,
                input=layer0_input,
                image_shape=(batch_size, 1, 28, 28),  # 入力画像のサイズを4Dテンソルで指定
                filter_shape=(20, 1, 5, 5),           # フィルタのサイズを4Dテンソルで指定
                poolsize=(2, 2))

    # layer0の出力がlayer1への入力となる
    # layer0の出力画像のサイズは (batch_size, 20, 12, 12)
    # 12x12ピクセルの画像が特徴マップ数分（20枚）ある
    # 畳み込みによって画像サイズは12x12ピクセルから8x8ピクセルに落ちる
    # プーリングによって画像サイズはさらに4x4ピクセルに落ちる
    # 特徴マップ数は50枚でそれぞれの特徴マップのサイズは4x4ピクセル
    # 最終的にこの層の出力のサイズは (batch_size, 50, 4, 4) になる
    layer1 = LeNetConvPoolLayer(rng,
                input=layer0.output,
                image_shape=(batch_size, 20, 12, 12), # 入力画像のサイズを4Dテンソルで指定
                filter_shape=(50, 20, 5, 5),          # フィルタのサイズを4Dテンソルで指定
                poolsize=(2, 2))

    # 隠れ層への入力
    # 画像のピクセルをフラット化する
    # layer1の出力のサイズは (batch_size, 50, 4, 4) なのでflatten()によって
    # (batch_size, 50*4*4) = (batch_size, 800) になる
    layer2_input = layer1.output.flatten(2)

    # 全結合された隠れ層
    # 入力が800ユニット、出力が500ユニット
    layer2 = HiddenLayer(rng,
        input=layer2_input,
        n_in=50 * 4 * 4,
        n_out=500,
        activation=T.tanh)

    # 最終的な数字分類を行うsoftmax層
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # コスト関数を計算するシンボル
    cost = layer3.negative_log_likelihood(y)

    # index番目のテスト用ミニバッチを入力してエラー率を返す関数を定義
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        })

    # index番目のバリデーション用ミニバッチを入力してエラー率を返す関数を定義
    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        })

    # パラメータ
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # コスト関数の微分
    grads = T.grad(cost, params)

    # パラメータ更新式
    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

    # index番目の訓練バッチを入力し、パラメータを更新する関数を定義
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        })

    print "train model ..."

    # eary-stoppingのパラメータ
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0
    start_time = time.clock()

    epoch = 0
    done_looping = False

    fp1 = open("validation_error.txt", "w")
    fp2 = open("test_error.txt", "w")

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
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

                    # テストデータのエラー率も計算
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print "    epoch %i, minibatch %i/%i, test error of best model %f %%" % (epoch, minibatch_index + 1, n_train_batches, test_score * 100)
                    fp2.write("%d\t%f\n" % (epoch, test_score * 100))

            # patienceを超えたらループを終了
            if patience <= iter:
                done_looping = True
                break

    fp1.close()
    fp2.close()

    end_time = time.clock()
    print "Optimization complete."
    print "Best validation score of %f %% obtained at iteration %i, with test performance %f %%" % (best_validation_loss * 100.0, best_iter + 1, test_score * 100.0)
    print "The code for file " + os.path.split(__file__)[1] + " ran for %.2fm" % ((end_time - start_time) / 60.0)

    import cPickle
    cPickle.dump(layer0, open("layer0.pkl", "wb"))
    cPickle.dump(layer1, open("layer1.pkl", "wb"))

if __name__ == '__main__':
    evaluate_lenet5()
