#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

"""
2クラスロジスティック回帰をtheanoで実装
"""

def plot_data(X, y):
    # positiveクラスのデータのインデックス
    positive = [i for i in range(len(y)) if y[i] == 1]
    # negativeクラスのデータのインデックス
    negative = [i for i in range(len(y)) if y[i] == 0]

    plt.scatter(X[positive, 0], X[positive, 1], c='red', marker='o', label="positive")
    plt.scatter(X[negative, 0], X[negative, 1], c='blue', marker='o', label="negative")

if __name__ == "__main__":
    # 訓練データをロード
    data = np.genfromtxt("../data/ex2data1.txt", delimiter=",")
    data_x = data[:, (0, 1)]
    data_y = data[:, 2]

    # 訓練データ数
    m = len(data_y)

    # 訓練データをプロット
    plt.figure(1)
    plot_data(data_x, data_y)

    # 訓練データの1列目に1を追加
    data_x = np.hstack((np.ones((m, 1)), data_x))

    # 訓練データを共有変数にする
    X = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
    y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)

    # パラメータを共有変数にし、0で初期化
    # 訓練データに1を加えたのでバイアスもthetaに含めてしまう
    theta = theano.shared(np.zeros(3, dtype=theano.config.floatX), name='theta', borrow=True)

    # コスト関数を定義
    # 交差エントロピー誤差関数
    h = T.nnet.sigmoid(T.dot(X, theta))
    cost = (1.0 / m) * T.sum(-y * T.log(h) - (1 - y) * T.log(1 - h))

    # 勾配降下法
    # コスト関数の微分
    g_theta = T.grad(cost=cost, wrt=theta)

    # パラメータ更新式
    learning_rate = 0.001
    updates = [(theta, theta - learning_rate * g_theta)]

    # 訓練用の関数を定義
    train_model = theano.function(inputs=[], outputs=cost, updates=updates)

    # 高度な収束判定はせずにiterations回だけ繰り返す
    iterations = 300000
    for iter in range(iterations):
        current_cost = train_model()
        print iter, current_cost

    # 更新されたパラメータを表示
    t = theta.get_value()
    print "theta:", t

    # 決定境界を描画
    plt.figure(1)
    xmin, xmax = min(data_x[:,1]), max(data_x[:,1])
    xs = np.linspace(xmin, xmax, 100)
    ys = [- (t[0] / t[2]) - (t[1] / t[2]) * x for x in xs]
    plt.plot(xs, ys, 'b-', label="decision boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim((30, 100))
    plt.ylim((30, 100))
    plt.legend()
    plt.show()
