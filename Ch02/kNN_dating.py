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


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    # 标0表示取每列最小/大，标1表示取每行最小/大，这里实质上是min-max归一化
    minVals = dataSet.min(0)
    maxVlas = dataSet.max(0)
    ranges = maxVlas - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet


# min-max归一化更易懂的形式
def autoNorm2(dataSet):
    # 标0表示取每列最小/大，标1表示取每行最小/大，这里实质上是min-max归一化
    minVals = dataSet.min(0)
    maxVlas = dataSet.max(0)
    ranges = maxVlas - minVals
    normDataSet = (dataSet - minVals) / ranges

    return normDataSet


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat = autoNorm(datingDataMat)
    # 获取数据条数
    m = normMat.shape[0]
    # 获取测试数据条数
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 使用0-99条测试，使用100-999构建数据
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
datingDataMat1 = autoNorm(datingDataMat)
# datingDataMat2 = autoNorm2(datingDataMat)[0]


if __name__ == '__main__':
    datingClassTest()
