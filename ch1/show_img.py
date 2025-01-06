import matplotlib.pyplot as plt
import matplotlib.image as ig

#환경에 따라 저장 디렉토리 위치 다름. 돌리고 싶음 수정하셈.
img = ig.imread('D:/Github/deeplearning_python/dataset/cactus.png')

plt.imshow(img)
plt.show()