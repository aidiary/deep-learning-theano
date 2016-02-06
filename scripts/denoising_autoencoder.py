#coding: utf-8
import time
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import gzip
import cPickle
from autoencoder import Autoencoder, load_data

class DenoisingAutoencoder(Autoencoder):
    """雑音除去自己符号化器"""
    def __init__(self, numpy_rng, theano_rng=None,
                 input=None,
                 n_visible=784, n_hidden=500,
                 W=None, bhid=None, bvis=None):
        Autoencoder.__init__(self, numpy_rng, theano_rng,
                             input, n_visible, n_hidden,
                             W, bhid, bvis)

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1-corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_cost_updates(self, corruption_level, learning_rate):
        """コスト関数と更新式のシンボルを返す"""
        # 入力の一部にノイズを付与して汚す
        tilde_x = self.get_corrupted_input(self.x, corruption_level)

        # 入力を変換
        y = self.get_hidden_values(tilde_x)

        # 変換した値を逆変換で入力に戻す
        z = self.get_reconstructed_input(y)

        # コスト関数のシンボル
        # 汚した入力が汚す前の入力に近くなるように学習する
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

        # Lはミニバッチの各サンプルの交差エントロピー誤差なので全サンプルで平均を取る
        cost = T.mean(L)

        # 誤差関数の微分
        gparams = T.grad(cost, self.params)

        # 更新式のシンボル
        updates = [(param, param - learning_rate * gparam)
                   for param, gparam in zip(self.params, gparams)]

        return cost, updates

    def __getstate__(self):
        """パラメータの状態を返す"""
        return (self.W.get_value(), self.b.get_value(), self.b_prime.get_value())

    def __setstate__(self, state):
        """パラメータの状態をセット"""
        self.W.set_value(state[0])
        self.b.set_value(state[1])
        self.b_prime.set_value(state[2])

if __name__ == "__main__":
    corruption_level = 0.3
    learning_rate = 0.1
    training_epochs = 20
    batch_size = 20

    # 学習データのロード
    datasets = load_data('mnist.pkl.gz')
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

    dA = DenoisingAutoencoder(numpy_rng=rng,
                              theano_rng=theano_rng,
                              input=x,
                              n_visible=28 * 28,
                              n_hidden=500)

    # コスト関数と更新式のシンボルを取得
    cost, updates = dA.get_cost_updates(corruption_level, learning_rate)

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
    f = open('dA.pkl', 'wb')
    cPickle.dump(dA.__getstate__(), f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()