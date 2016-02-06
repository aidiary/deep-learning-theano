#coding: utf-8
import time
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import gzip
import cPickle

class Autoencoder(object):
    def __init__(self, numpy_rng, theano_rng=None,
                 input=None,
                 n_visible=784, n_hidden=500,
                 W=None, bhid=None, bvis=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            # 入力層と出力層の間の重み
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6.0 / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6.0 / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            # 入力層（visible）のユニットのバイアス
            bvis = theano.shared(
                value=np.zeros(n_visible, dtype=theano.config.floatX),
                borrow=True)

        if not bhid:
            # 隠れ層（hidden）のユニットのバイアス
            bhid = theano.shared(
                value=np.zeros(n_hidden, dtype=theano.config.floatX),
                name='b',
                borrow=True)

        # パラメータ
        self.W = W
        self.b = bhid
        self.W_prime = self.W.T
        self.b_prime = bvis
        self.params = [self.W, self.b, self.b_prime]

        self.theano_rng = theano_rng

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

    def get_hidden_values(self, input):
        """入力層の値を隠れ層の値に変換"""
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """隠れ層の値を入力層の値に逆変換"""
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, learning_rate):
        """コスト関数と更新式のシンボルを返す"""
        # 入力を変換
        y = self.get_hidden_values(self.x)

        # 変換した値を逆変換で入力に戻す
        z = self.get_reconstructed_input(y)

        # コスト関数のシンボル
        # 元の入力と再構築した入力の交差エントロピー誤差を計算
        # 入力xがミニバッチのときLはベクトルになる
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

        # Lはミニバッチの各サンプルの交差エントロピー誤差なので全サンプルで平均を取る
        cost = T.mean(L)
#         cost += 0.001 * abs(self.W).sum()    # L1 regularization
#         cost += 0.001 * (self.W ** 2).sum()  # L2 regularization

        # 誤差関数の微分
        gparams = T.grad(cost, self.params)

        # 更新式のシンボル
        updates = [(param, param - learning_rate * gparam)
                   for param, gparam in zip(self.params, gparams)]

        return cost, updates

    def feedforward(self):
        """入力をフィードフォワードさせて出力を計算"""
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        return z

    def __getstate__(self):
        """パラメータの状態を返す"""
        return (self.W.get_value(), self.b.get_value(), self.b_prime.get_value())

    def __setstate__(self, state):
        """パラメータの状態をセット"""
        self.W.set_value(state[0])
        self.b.set_value(state[1])
        self.b_prime.set_value(state[2])

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

if __name__ == "__main__":
    learning_rate = 0.1
    training_epochs = 20
    batch_size = 20

    # 学習データのロード
    datasets = load_data('../data/mnist.pkl.gz')
    # 自己符号化器は教師なし学習なので訓練データのラベルは使わない
    train_set_x = datasets[0][0]

    # ミニバッチ数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # ミニバッチのインデックスを表すシンボル
    index = T.lscalar()

    # ミニバッチの学習データを表すシンボル
    x = T.matrix('x')

    # モデル構築
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    autoencoder = Autoencoder(numpy_rng=rng,
                               theano_rng=theano_rng,
                               input=x,
                               n_visible=28 * 28,
                               n_hidden=500)

    # コスト関数と更新式のシンボルを取得
    cost, updates = autoencoder.get_cost_updates(learning_rate=learning_rate)

    # 訓練用の関数を定義
    train_da = theano.function([index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        })

    # モデル訓練
    fp = open("cost.txt", "w")
    start_time = time.clock()
    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print "Training epoch %d, cost %f" % (epoch, np.mean(c))
        fp.write("%d\t%f\n" % (epoch, np.mean(c)))
        fp.flush()

    end_time = time.clock()
    training_time = (end_time - start_time)
    fp.close()

    print "time: %ds" % (training_time)

    # 学習したモデルの状態を保存
    f = open('autoencoder.pkl', 'wb')
    cPickle.dump(autoencoder.__getstate__(), f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()