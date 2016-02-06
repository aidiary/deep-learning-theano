#coding:utf-8
import numpy as np
import theano
from theano.tensor.shared_randomstreams import RandomStreams

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

# 二項分布だが試行回数n=1なのでいわゆるベルヌーイ分布
# ベルヌーイ分布からサンプルを (10, 10) 生成する
corruption_level = 0.5
binomial = theano.function([], theano_rng.binomial(size=(10, 10), n=1, p=1-corruption_level))
print binomial()
