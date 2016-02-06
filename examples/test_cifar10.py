#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# CIFAR-10
# 80 million tiny imagesのサブセット
# Alex Krizhevsky, Vinod Nair, Geoffrey Hintonが収集
# 32x32のカラー画像60000枚
# 10クラスで各クラス6000枚
# 50000枚の訓練画像と10000枚（各クラス1000枚）のテスト画像
# クラスラベルは排他的
# PythonのcPickle形式で提供されている

def unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

# ラベル名をロード
label_names = unpickle("cifar10/batches.meta")["label_names"]
d = unpickle("cifar10/data_batch_1")
data = d["data"]
labels = np.array(d["labels"])
nsamples = len(data)

print label_names

# 各クラスの画像をランダムに10枚抽出して描画
nclasses = 10
pos = 1
for i in range(nclasses):
    # クラスiの画像のインデックスリストを取得
    targets = np.where(labels == i)[0]
    np.random.shuffle(targets)
    # 最初の10枚の画像を描画
    for idx in targets[:10]:
        plt.subplot(10, 10, pos)
        img = data[idx]
        # (channel, row, column) => (row, column, channel)
        plt.imshow(img.reshape(3, 32, 32).transpose(1, 2, 0))
        plt.axis('off')
        label = label_names[i]
        pos += 1
plt.show()
