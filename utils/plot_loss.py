#coding: utf-8
import matplotlib.pyplot as plt

epoch = []
loss = []

fp = open("sa_validation_error.txt", "r")
fp.readline()
for line in fp:
    epc, acc = line.rstrip().split()
    epoch.append(int(epc))
    loss.append(float(acc))
fp.close()

plt.plot(epoch, loss, 'o-')

plt.title("stacked autoencoder")
plt.xlabel("epoch")
plt.ylabel("validation error")
plt.legend(loc="best", ncol=1, fontsize=10, numpoints=1)
plt.grid()
plt.tight_layout()
plt.show()
