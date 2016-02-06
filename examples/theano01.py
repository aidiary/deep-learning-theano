#coding: utf-8
import theano
import theano.tensor as T

# シンボルの生成
# xはdoubleのスカラー型
x = T.dscalar('x')
print type(x)

# シンボルを組み立てて数式を定義（これもまたシンボル）
y = x ** 2
print type(y)

# シンボルを使って関数を定義
# ここでコンパイルされる
f = theano.function(inputs=[x], outputs=y)
print type(f)

# 関数を使ってxに具体的な値を入れてyを計算
print f(1)
print f(2)
print f(3)
print f(4)
print f(5)
