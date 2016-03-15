#coding: utf-8
import os
import timeit
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data

class RBM(object):
    """制約ボルツマンマシン (Restricted Boltzmann Machine: RBM)"""
    def __init__(self, input=None,
                 n_visible=784, n_hidden=500,
                 w=None, hbias=None, vbias=None,
                 numpy_rng=None, theano_rng=None):
        # 可視層のユニット数
        self.n_visible = n_visible
        # 隠れ層のユニット数
        self.n_hidden = n_hidden

        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # 重みを表す共有変数を作成
        if W is None:
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6.0 / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6.0 / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        # 隠れ層のバイアスを表す共有変数を作成
        if hbias is None:
            hbias = theano.shared(
                value=np.zeros(n_hidden, dtype=theano.config.floatX),
                name='hbias',
                borrow=True)

        # 可視層のバイアスを表す共有変数を作成
        if vbias is None:
            vbias = theano.shared(
                value=np.zeros(n_visible, dtype=thenao.config.floatX),
                name='vbias',
                borrow=True)

        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng

        # 学習対象となるパラメータリスト
        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        """自由エネルギーを計算するシンボルを返す"""
        vbias_term = T.dot(v_sample, self.vbias)
        wx_b = T.dot(v_sample, self.W) + self.hbias
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return - hidden_term - vbias_term

    def propup(self, vis):
        """可視層を引数で固定したときの隠れ層の確率 P(hi=1|v) を
        計算するシンボルを返す
        シグモイドを適用する前の式も合わせて返す"""
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation,
                T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        """P(hi=1|v) の分布に基づいて隠れ層の各ユニットの値を
        サンプリングするシンボルを返す"""
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        """隠れ層を引数で固定した時の可視層の確率 P(vj=1|h) を
        計算するシンボルを返す
        シグモイドを適用する前の式も合わせて返す"""
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation,
                T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        """P(vj=1|h) の分布に基づいて可視層の各ユニットの値を
        サンプリングするシンボルを返す"""
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        """隠れ層の値から始まるGibbsサンプリングの1ステップ分
        h0_sample => v1_sample => h1_sample"""
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        """可視層の値から始まるGibbsサンプリングの1ステップ分
        v0_sample => h1_sample => v1_sample"""
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, k=1):
        """RBMのコスト関数と更新式のシンボルを返す"""
        # CD法は訓練データ（self.input）からサンプリングを開始
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        chain_start = ph_sample

        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k)

        # k回のサンプルのうち必要なのは可視層の最後のサンプル
        chain_end = nv_samples[-1]

        # コスト関数
        # 第2項はCD法による近似
        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))

        # コスト関数の各パラメータでの勾配
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # SGDによる更新式
        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)

        # RBMではコスト関数をそのままモニタリングしない（TODO:なぜcostではダメなのか？）
        # v => h => v'での (v, v') の交差エントロピーを再構築誤差として返す
        # hとv'はサンプルではなく確率の方を使う
        # denoising autoencoderと同じ
        # Tutorialではupdateも引数としているが未使用変数なので削除
        monitoring_cost = self.get_reconstruction_cost(pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_reconstruction_cost(self, pre_sigmoid_nv):
        """再構築誤差を返す"""
        # sigmoidを取る前の値はここで必要
        # log(sigmoid(x)) でnanにならないようにするため？
        # Tutorialでは先頭に - がついていないけどバグ？
        L = - T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                    (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                    axis=1)
        cross_entropy = T.mean(L)

        return cross_entropy


def test_rbm(learning_rate=0.1, training_epochs=15,
             dataset='mnist.pkl.gz', batch_size=20,
             n_chains=20, n_samples=10, output_dir='rbm_plots',
             n_hidden=500):
    dataset = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    index = T.lscalar()
    x = T.matrix('x')

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    rbm = RBM(input=x, n_visible=28 * 28,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate, k=15)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)

    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index*batch_size: (index + 1)*batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.0
    start_time = timeit.default_timer()

    for epoch in range(training_epochs):
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print 'Training epoch %d, cost is ' % epoch, np.mean(mean_cost)

        # plot filters after each training epoch
        plotting_start = timeit.default_timer()
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tiem_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plogging_start)

    end_time = timeit.default_timer()
    pretraining_time = (end_time - start_time) - plotting_time

    print 'Training took %f minutes' % (pretraining_time / 60.0)

    # Sampling from RBM
