#coding: utf-8
import numpy as np
import theano

# データを共有変数に読み込む
data = np.array([[1,2,3], [4,5,6]], dtype=theano.config.floatX)
X = theano.shared(data, name='X', borrow=True)
print type(X)
print X.get_value()
