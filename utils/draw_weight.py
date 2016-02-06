%matplotlib inline
import numpy as np
import cPickle
import matplotlib.pyplot as plt
model = cPickle.load(open("cifar10.pkl", "rb"))

n1, n2, h, w = model.conv1.W.shape
print n1, n2, h, w
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
pos = 0
for i in range(4):
    for j in range(8):
        ax = fig.add_subplot(4, 8, pos + 1, xticks=[], yticks=[])
        img = model.conv1.W[pos].transpose(1, 2, 0)
        img -= img.min()
        img /= img.max()
        print img
        ax.imshow(img)
        pos += 1