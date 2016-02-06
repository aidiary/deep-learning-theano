#coding: utf-8
import cPickle
import gzip
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        """ロジスティック回帰モデルの初期化
        input: ミニバッチ単位のデータ行列（n_samples, n_in）
        n_in : 入力の次元数
        n_out: 出力の次元数"""
        # 重み行列を初期化
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                               name='W',
                               borrow=True)

        # バイアスベクトルを初期化
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX),
                               name='b',
                               borrow=True)


        # 各サンプルが各クラスに分類される確率を計算するシンボル
        # 全データを行列化してまとめて計算している
        # 出力は(n_samples, n_out)の行列
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # 確率が最大のクラスのインデックスを計算
        # 出力は(n_samples,)のベクトル
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # ロジスティック回帰モデルのパラメータ
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """誤差関数である負の対数尤度を計算するシンボルを返す
        yにはinputに対応する正解クラスを渡す"""
        # 式通りに計算するとsumだがmeanの方がよい
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """分類の誤差率を計算するシンボルを返す
        yにはinputに対応する正解クラスを渡す"""
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def load_data(dataset):
    """データセットをロードしてGPUの共有変数に格納"""
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy

        # 共有変数には必ずfloat型で格納
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

        # ラベルはint型なのでキャストして返す
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, batch_size=600):
    # 学習データの準備
    datasets = load_data('../data/mnist.pkl.gz')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # ミニバッチの数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print "building the model ..."

    # シンボルの割り当て
    # ミニバッチのインデックスを表すシンボル
    index = T.lscalar()

    # ミニバッチの学習データとラベルを表すシンボル
    x = T.matrix('x')
    y = T.ivector('y')

    # MNISTの手書き数字を分類するロジスティック回帰モデル
    # 入力は28ピクセルx28ピクセルの画像、出力は0から9のラベル
    # 入力はシンボルxを割り当てておいてあとで具体的なデータに置換する
    classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)

    # シンボルのサイズを知りたいとき
    # シンボルに具体的なデータを与えて評価しないと取得できない
#     get_shape = theano.function([index], classifier.p_y_given_x.shape,
#                         givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})
#     print get_shape(0)

    # 誤差（コスト）を計算 => 最小化したい
    cost = classifier.negative_log_likelihood(y)

    # index番目のテスト用ミニバッチを入力してエラー率を返す関数を定義
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={  # ここで初めてシンボル x, y を具体的な値で置き換える
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
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # パラメータ更新式
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # index番目の訓練バッチを入力し、パラメータを更新する関数を定義
    # 戻り値としてコストが返される
    # この関数の呼び出し時にindexに具体的な値が初めて渡される
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        })

    # モデル訓練
    print 'training the model ...'

    # eary-stoppingのパラメータ
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    test_score = 0
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            # minibatch_index番目の訓練データのミニバッチを用いてパラメータ更新
            minibatch_avg_cost = train_model(minibatch_index)

            # validation_frequency回の更新ごとにバリデーションセットによるモデル検証が入る
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                # バリデーションセットの平均エラー率を計算
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print "epoch %i, minibatch %i/%i, validation error %f %%" % (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100)

                # エラー率が十分改善したならまだモデル改善の余地があるためpatienceを上げてより多くループを回せるようにする
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        print "*** iter %d / patience %d" % (iter, patience)

                    best_validation_loss = this_validation_loss

                    # テストセットを用いたエラー率も求めておく
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print "    epoch %i, minibatch %i/%i, test error of best model %f %%" % (epoch, minibatch_index + 1, n_train_batches, test_score * 100)

            # patienceを超えたらループを終了
            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print "Optimization complete with best validation score of %f %%, with test performance %f %%" % (best_validation_loss * 100, test_score * 100)
    print "The code run for %d epochs, with %f epochs/sec" % (epoch, 1.0 * epoch / (end_time - start_time))

if __name__ == "__main__":
    sgd_optimization_mnist()
