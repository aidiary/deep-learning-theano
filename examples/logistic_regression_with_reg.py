#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

"""
2クラスロジスティック回帰（非線形分離の例）をtheanoで実装
"""

def plot_data(X, y):
    # positiveクラスのデータのインデックス
    positive = [i for i in range(len(y)) if y[i] == 1]
    # negativeクラスのデータのインデックス
    negative = [i for i in range(len(y)) if y[i] == 0]

    plt.scatter(X[positive, 0], X[positive, 1], c='red', marker='o', label="positive")
    plt.scatter(X[negative, 0], X[negative, 1], c='blue', marker='o', label="negative")

def mapFeature(x1, x2, degree=6):
    """
    特徴x1と特徴x2を組み合わせたdegree次の項まで特徴をデータに追加
    バイアス項に対応するデータ1も追加
    """
    # データ行列に1を追加
    m = x1.shape[0]
    data_x = np.ones((m, 1))
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            new_x = (x1 ** (i - j) * x2 ** j).reshape((m, 1))
            data_x = np.hstack((data_x, new_x))
    return data_x

if __name__ == "__main__":
    # 訓練データをロード
    data = np.genfromtxt("../data/ex2data2.txt", delimiter=",")
    data_x = data[:, (0, 1)]
    data_y = data[:, 2]

    # 訓練データ数
    m = len(data_y)

    # 訓練データをプロット
    plt.figure(1)
    plot_data(data_x, data_y)

    # 特徴量のマッピング
    # 元の特徴量の6次までの多項式項を追加
    # 1列目の1も追加する
    data_x = mapFeature(data_x[:, 0], data_x[:, 1], 6)

    # 訓練データを共有変数にする
    X = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
    y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)

    # パラメータを共有変数にし、0で初期化
    # 訓練データに1を加えたのでバイアスもthetaに含めてしまう
    theta = theano.shared(np.zeros(data_x.shape[1], dtype=theano.config.floatX), name='theta', borrow=True)

    # コスト関数を定義
    # 交差エントロピー誤差関数
    lam = 1.0
    h = T.nnet.sigmoid(T.dot(X, theta))
    cost = (1.0 / m) * T.sum(-y * T.log(h) - (1 - y) * T.log(1 - h)) + (lam / (2 * m)) * T.sum(theta ** 2)

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
    gridsize = 100
    x1_vals = np.linspace(-1, 1.5, gridsize)
    x2_vals = np.linspace(-1, 1.5, gridsize)
    x1_vals, x2_vals = np.meshgrid(x1_vals, x2_vals)
    z = np.zeros((gridsize, gridsize))
    for i in range(gridsize):
        for j in range(gridsize):
            x1 = np.array([x1_vals[i, j]])
            x2 = np.array([x2_vals[i, j]])
            z[i, j] = np.dot(mapFeature(x1, x2), theta.get_value())

    # 決定境界はsigmoid(z)=0.5、すなわちz=0の場所
    plt.contour(x1_vals, x2_vals, z, levels=[0])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim((-1, 1.5))
    plt.ylim((-1, 1.5))
    plt.legend()
    plt.show()
