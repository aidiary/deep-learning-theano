#coding: utf-8
import theano
import theano.tensor as T

# シンボルを定義
x = T.dmatrix('x')

# シンボルを組み合わせて数式を定義
# s = T.nnet.sigmoid(x)でもOK
s = 1 / (1 + T.exp(-x))

# シンボルを使って関数化
sigmoid = theano.function(inputs=[x], outputs=s)

# 実際の値を使って計算
print sigmoid([[0, 1], [-1, -2]])
