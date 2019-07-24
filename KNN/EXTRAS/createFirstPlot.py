from numpy import *

from KNN import kNN_dating
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat, datingLabels = kNN_dating.file2matrix('../datingTestSet2.txt')
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
# 带颜色，用第0列和第1列表示效果较好
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# ax.axis([-2,25,-0.2,2.0])
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
plt.show()



