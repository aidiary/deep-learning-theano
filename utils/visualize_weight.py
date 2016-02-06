#coding: utf-8
import numpy as np
import cPickle
import matplotlib.pyplot as plt

def visualize_weights(W, outfile):
    # 重みをスケーリング
    W = (W - np.min(W)) / (np.max(W) - np.min(W))
    W *= 255.0
    W = W.astype(np.int)

    pos = 1
    for i in range(100):
        plt.subplot(10, 10, pos)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(W[i].reshape(28, 28))
        plt.gray()
        plt.axis('off')
        pos += 1
    plt.show()
    plt.savefig(outfile)


if __name__ == "__main__":
    # ファイルから学習したパラメータをロード
    f = open("autoencoder.pkl", "rb")
    state = cPickle.load(f)
    f.close()

    # 重みを取り出す
    W = state[0]

    # 学習した重みを可視化
    # Wを転置しているのはサンプルを行方向にするため
    visualize_weights(W.T, "autoencoder_filters.png")
