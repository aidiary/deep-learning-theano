#coding:utf-8
import numpy as np
import theano
import theano.tensor as T

x = T.dmatrix()
z = T.dmatrix()

L = T.sum(x * z, axis=1)
cost = T.mean(L)

f1 = theano.function([x, z], L)
f2 = theano.function([x, z], cost)

# 実際の値を代入して計算
x_value = np.array([1,2,3,4,5,6]).reshape((3, 2))
z_value = np.array([10,20,30,40,50,60]).reshape((3, 2))
print f1(x_value, z_value)
print f2(x_value, z_value)
