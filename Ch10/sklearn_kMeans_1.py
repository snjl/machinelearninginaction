import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

def loadDataSet(filename):
    '''
    读取数据集

    Args:
        filename: 文件名
    Returns:
        dataMat: 数据样本矩阵
    '''
    dataMat = []
    with open(filename, 'rb') as f:
        for line in f:
            # 读取的字节流需要先解码成utf-8再处理
            eles = list(map(float, line.decode('utf-8').strip().split('\t')))
            dataMat.append(eles)
    return dataMat


dataMat = np.array(loadDataSet('testSet.txt'))
m, n = np.shape(dataMat)

set_k = 4

y_pred = KMeans(n_clusters=set_k).fit_predict(dataMat)

fig = plt.figure()
title = 'kmeans with k={}'.format(set_k)
ax = fig.add_subplot(111, title=title)

plt.scatter(dataMat[:, 0], dataMat[:, 1],c=y_pred)

plt.show()


