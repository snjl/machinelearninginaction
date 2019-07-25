import numpy as np
import matplotlib.pyplot as plt
import operator


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


"""
函数说明:kNN算法,分类器

Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    classCount[0][0] - 分类结果
"""


def classify0(inX, dataSet, labels, k):
    # 广播，每个样本均减去inX
    diff = dataSet - inX
    # 算出减去后的平方值
    distances = diff * diff
    # 相加后求平方根，即算出距离（实际上仅相加即可，因为这是一个增函数）
    distances = distances.sum(axis=1) ** 0.5
    # 按距离从小到大排序，求出索引值
    distances_min_index = distances.argsort()
    classCount = dict()
    # 如果k过于大，则以dataSet的长度为准，取出前k个元素的类别
    k = min(len(dataSet), k)
    for i in range(k):
        voteIlabel = labels[distances_min_index[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # reverse降序排列字典
    classCount = sorted(classCount.items(), key=lambda v: v[1], reverse=True)
    # 返回次数最多的类别
    return classCount[0][0]


groups, labels = createDataSet()

print(classify0([-2, -2], groups, labels, 5))
