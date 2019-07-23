https://github.com/EchoLLLiu/more-ML-algorithm


https://www.cnblogs.com/lliuye/p/9144312.html

# 文件备注
kMeans.py 机器学习实战中代码，包含kmeans和二分kmeans代码
kMeans_1.py 包含本文档写的kmeans代码

kmeans算法又名k均值算法。其算法思想大致为：先从样本集中随机选取 k 个样本作为簇中心，并计算所有样本与这 k 个“簇中心”的距离，对于每一个样本，将其划分到与其距离最近的“簇中心”所在的簇中，对于新的簇计算各个簇的新的“簇中心”。

根据以上描述，我们大致可以猜测到实现kmeans算法的主要三点：

（1）簇个数 k 的选择

（2）各个样本点到“簇中心”的距离

（3）根据新划分的簇，更新“簇中心”

## 算法过程

```md
输入：训练数据集 D=x(1),x(2),...,x(m) ,聚类簇数 k ;
  过程：函数 kMeans(D,k,maxIter) .
  1：从 D 中随机选择 k 个样本作为初始“簇中心”向量： μ(1),μ(2),...,,μ(k) :
  2：repeat
  3：  令 Ci=∅(1≤i≤k)
  4：  for j=1,2,...,m do
  5：    计算样本 x(j) 与各“簇中心”向量 μ(i)(1≤i≤k) 的欧式距离
  6：    根据距离最近的“簇中心”向量确定 x(j) 的簇标记： λj=argmini∈{1,2,...,k}dji
  7：    将样本 x(j) 划入相应的簇： Cλj=Cλj⋃{x(j)} ;
  8：  end for
  9：  for i=1,2,...,k do
  10：    计算新“簇中心”向量： (μ(i))′=1|Ci|∑x∈Cix ;
  11：    if (μ(i))′=μ(i) then
  12：      将当前“簇中心”向量 μ(i) 更新为 (μ(i))′
  13：    else
  14：      保持当前均值向量不变
  15：    end if
  16：  end for
  17：  else
  18：until 当前“簇中心”向量均未更新
  输出：簇划分 C=C1,C2,...,CK
  
```

为避免运行时间过长，通常设置一个最大运行轮数或最小调整幅度阈值，若达到最大轮数或调整幅度小于阈值，则停止运行。


```python
import numpy as np
import matplotlib.pyplot as plt


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


def distEclud(vecA, vecB):
    '''
    计算两向量的欧氏距离

    Args:
        vecA: 向量A
        vecB: 向量B
    Returns:
        欧式距离
    '''
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def kMeans(dataSet, k, maxIter=5):
    '''
    K-Means

    Args:
        dataSet: 数据集
        k: 聚类数
    Returns:
        centroids: 聚类中心
        clusterAssment: 点分配结果
    '''
    # 随机初始化聚类中心
    centroids = randCent(dataSet, k)
    init_centroids = centroids.copy()

    m, n = dataSet.shape

    # 点分配结果：第一列指明样本所在的簇，第二列指明该样本到聚类中心的距离
    clusterAssment = np.mat(np.zeros((m, 2)))

    # 标识聚类中心是否仍在变化
    clusterChanged = True

    # 直至聚类中心不再变化
    iterCount = 0
    while clusterChanged and iterCount < maxIter:
        iterCount += 1
        clusterChanged = False
        # 分配样本到簇
        for i in range(m):
            # 计算第i个样本到各个聚类中心的距离
            minIndex = 0
            minDist = np.inf
            for j in range(k):
                dist = distEclud(dataSet[i, :], centroids[j, :])
                if dist < minDist:
                    minIndex = j
                    minDist = dist
            # 任何一个样本的类簇分配发生变化则认为变化
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2

        # 刷新聚类中心：移动聚类中心点到所有簇的均值位置
        for cent in range(k):
            # 通过数组过滤得到簇中的点
            # matrix.A 是将matrix-->array
            # 通过np.nonzero(a==2)可以获得matrix a中为2的元素的横纵坐标，例如
            # >>> a = [[1,2],[2,3],[3,4],[5,6]]
            # >>> a = np.mat(a)
            # >>> np.nonzero(a==2)
            # >>> (array([1, 2], dtype=int64), array([1, 0], dtype=int64))
            # 其中，np.nonzero(a==2)[0]为所有横坐标，由于这里只需要横坐标值等于cent，得到横坐标后提取出来，在dataSet
            # 中找出后放入ptsInCluster，计算均值后存入centroids
            ptsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            if ptsInCluster.shape[0] > 0:
                # 计算均值并移动
                centroids[cent, :] = np.mean(ptsInCluster, axis=0)
    return centroids, clusterAssment, init_centroids


def randCent(dataSet, k):
    '''
    随机生成k个聚类中心

    Args:
        dataSet: 数据集
        k: 簇数目
    Returns:
        centroids: 聚类中心矩阵
    '''
    m, _ = dataSet.shape
    # 随机从数据集中选几个作为初始聚类中心
    centroids = dataSet.take(np.random.choice(80, k), axis=0)
    return centroids


dataMat = np.mat(loadDataSet('data/testSet.txt'))
m, n = np.shape(dataMat)

set_k = 4
centroids, clusterAssment, init_centroids = kMeans(dataMat, set_k)

clusterCount = np.shape(centroids)[0]
# 我们这里只设定了最多四个簇的样式，所以前面如果set_k设置超过了4，后面就会出现index error
patterns = ['o', 'D', '^', 's']
colors = ['b', 'g', 'y', 'black']

fig = plt.figure()
title = 'kmeans with k={}'.format(set_k)
ax = fig.add_subplot(111, title=title)
for k in range(clusterCount):
    # 绘制聚类中心
    ax.scatter(centroids[k, 0], centroids[k, 1], color='r', marker='+', linewidth=20)
    # 绘制初始聚类中心
    ax.scatter(init_centroids[k, 0], init_centroids[k, 1], color='purple', marker='*', linewidth=10)
    # 绘制属于该聚类中心的样本
    ptsInCluster = dataMat[np.nonzero(clusterAssment[:, 0].A == k)[0]]
    ax.scatter(ptsInCluster[:, 0].flatten().A[0], ptsInCluster[:, 1].flatten().A[0], color=colors[k],
               marker=patterns[k])

plt.show()


```