#coding: utf-8
import numpy as np
import cPickle
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from autoencoder import Autoencoder, load_data
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # テストに使うデータミニバッチ
    x = T.matrix('x')

    # ファイルから学習したパラメータをロード
    f = open("autoencoder.pkl", "rb")
    state = cPickle.load(f)
    f.close()

    # 自己符号化器を構築
    # 学習時と同様の構成が必要
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    autoencoder = Autoencoder(numpy_rng=rng,
                              theano_rng=theano_rng,
                              input=x,
                              n_visible=28*28,
                              n_hidden=500)

    # 学習したパラメータをセット
    autoencoder.__setstate__(state)

    # テスト用データをロード
    # 訓練時に使わなかったテストデータで試す
    datasets = load_data('mnist.pkl.gz')
    test_set_x = datasets[2][0]

    # 最初の100枚の画像を描画
    # test_set_xは共有変数なのでget_value()で内容を取得できる
    pos = 1
    for i in range(100):
        plt.subplot(10, 10, pos)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(test_set_x.get_value()[i].reshape(28, 28))
        plt.gray()
        plt.axis('off')
        pos += 1
    plt.savefig("original_image.png")

    # 最初の100枚のテスト画像を入力して再構築した画像を得る関数を定義
    feedforward = theano.function([],
        autoencoder.feedforward(),  # 出力を返すシンボル
        givens={ x: test_set_x[0:100] })

    # test_set_xのミニバッチの出力層の出力を計算
    output = feedforward()

    print output.shape

    # 出力は0-1に正規化されているため0-255のピクセル値に戻す
    output *= 255.0
    output = output.astype(np.int)

    # 画像を描画
    pos = 1
    for i in range(100):
        plt.subplot(10, 10, pos)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(output[i].reshape(28, 28))
        plt.gray()
        plt.axis('off')
        pos += 1
    plt.savefig("reconstructed_image.png")
