#coding: utf-8
import theano
import theano.tensor as T
import numpy as np

x = T.dscalar('x')
y = T.dscalar('y')

# 微分される数式のシンボルを定義
z = (x + 2 * y) ** 2

# zをxに関して偏微分
gx = T.grad(cost=z, wrt=x)

# zをyに関して偏微分
gy = T.grad(cost=z, wrt=y)

# 微分係数を求める関数を定義
fgx = theano.function(inputs=[x, y], outputs=gx)
fgy = theano.function(inputs=[x, y], outputs=gy)
print theano.pp(fgx.maker.fgraph.outputs[0])
print theano.pp(fgy.maker.fgraph.outputs[0])

# 具体的な値を与えて偏微分係数を求める
print fgx(1, 2)
print fgx(2, 2)
print fgy(1, 2)
print fgy(2, 2)
