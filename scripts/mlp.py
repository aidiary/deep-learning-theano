#coding: utf-8
import os
import time
import numpy as np
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """隠れ層の初期化
        rng: 乱数生成器（重みの初期化で使用）
        input: ミニバッチ単位のデータ行列（n_samples, n_in)
        n_in: 入力データの次元数
        n_out: 隠れ層のユニット数
        W: 隠れ層の重み
        b: 隠れ層のバイアス
        activation: 活性化関数
        """
        self.input = input

        # 隠れ層の重み（共有変数）を初期化（[Xavier10]による）
        if W is None:
            W_values = np.asarray(
                rng.uniform(low=-np.sqrt(6.0 / (n_in + n_out)),
                            high=np.sqrt(6.0 / (n_in + n_out)),
                            size=(n_in, n_out)),
                dtype=theano.config.floatX)  # @UndefinedVariable
            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        # 隠れ層のバイアス（共有変数）を初期化
        if b is None:
            b_values = np.zeros((n_out, ), dtype=theano.config.floatX)  # @UndefinedVariable
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # 隠れ層の出力を計算するシンボルを作成
        lin_output = T.dot(input, self.W) + self.b
        if activation is None:  # 線形素子の場合
            self.output = lin_output
        else:  # 非線形な活性化関数を通す場合
            self.output = activation(lin_output)

        # 隠れ層のパラメータ
        self.params = [self.W, self.b]

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        # 多層パーセプトロンは隠れ層とロジスティック回帰で表される出力層から成る
        # 隠れ層の出力がロジスティック回帰の入力になる点に注意
        self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)
        self.logRegressionLayer = LogisticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)

        # L1/L2正則化の正則化項を計算するシンボル
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()

        # MLPの誤差関数を計算するシンボル
        # 出力層にのみ依存するのでロジスティック回帰の実装と同じでよい
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        # 誤差を計算するシンボル
        self.errors = self.logRegressionLayer.errors

        # 多層パーセプトロンのパラメータ
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
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

    # 乱数生成器
    rng = np.random.RandomState(1234)

    # MLPを構築
    classifier = MLP(rng=rng, input=x, n_in=28 * 28, n_hidden=n_hidden, n_out=10)

    # コスト関数を計算するシンボル
    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    # index番目のテスト用ミニバッチを入力してエラー率を返す関数を定義
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        })

    # index番目のバリデーション用ミニバッチを入力してエラー率を返す関数を定義
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        })

    # コスト関数の各パラメータでの微分を計算
    gparams = [T.grad(cost, param) for param in classifier.params]

    # パラメータ更新式
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)]

    # index番目の訓練バッチを入力し、パラメータを更新する関数を定義
    # 戻り値としてコストが返される
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        })

    print "training the model ..."

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
            minibatch_avg_cost = train_model(minibatch_index)

            iter = (epoch - 1) * n_train_batches + minibatch_index
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
    print "Optimization complete. Best validation score of %f %% obtained at iteration %i, with test performance %f %%" % (best_validation_loss * 100.0, best_iter + 1, test_score * 100.0)
    print "The code for file " + os.path.split(__file__)[1] + " ran for %.2fm" % ((end_time - start_time) / 60.0)

if __name__ == "__main__":
    test_mlp(dataset="../data/mnist.pkl.gz")
