# 파이썬3 버전 이전에는 cPickle을 사용해야 했습니다.
# import cPickle
import pickle
import matplotlib.pyplot as plt
model = pickle.load(open("model.pkl", "rb"))
print(model.conv1.W.shape)

n1, n2, h, w = model.conv1.W.shape
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(n1):
    ax = fig.add_subplot(2, 10, i+1, wticks=[], yticks=[])
    ax.imshow(model.conv1.W[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()