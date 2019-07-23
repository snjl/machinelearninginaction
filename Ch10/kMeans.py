from numpy import *


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(list(fltLine))
    return dataMat


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    # 获得数据量m
    m = shape(dataSet)[0]
    # 聚到某类和距离记录，[类别,距离]
    clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points 
    # to a centroid, also holds SE of each point
    # 初始化聚类中心，可更换函数
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 如果第i个数据的类别不是当前距离最小类别，则聚类发生改变
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        # 聚类中心更改
        for cent in range(k):  # recalculate centroids
            # 提取现在每个类别的横坐标，并且提取出dataSet中的值
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            # mean(x,axis=0)列平均
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean 
    return centroids, clusterAssment


# 测试聚类
def main_test_1():
    dataMat = loadDataSet('testSet.txt')
    dataMat = mat(dataMat)
    kMeans(dataMat, 4)


def biKmeans(dataSet, k, distMeas=distEclud):
    # 获得数据量m
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    # 第一个聚类中心生成
    centroid0 = mean(dataSet, axis=0)
    print('first center:', centroid0)
    centList = [centroid0,]
    # 计算聚类中心与所有点的距离（此处使用了平方）
    for j in range(m):  # calc initial Error
        clusterAssment[j, 1] = distMeas(centroid0, dataSet[j, :]) ** 2
    while len(centList) < k:
        lowestSSE = inf
        # 计算已划分的第i簇，假设划分为2簇后的SSE
        for i in range(len(centList)):
            # 取出聚类为i的点
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 对第i类进行k=2的聚类，返回聚类中心点和类别-距离信息
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 计算第i类划分成2簇后的总SSE，记作sseSplit
            sseSplit = sum(splitClustAss[:, 1])  # compare the SSE to the currrent minimum
            # 计算当前未使用来划分的簇的SSE
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            # 两个SSE加起来表示使用第i类进行kmeans(k=2)划分后的SSE
            # 如果已经到达最低值，则使用该种划分，记录下所有值
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 重新编号（由于kmeans求出来的聚类为0,1，所以将聚类中的1重新编号为len(centList),聚类中的0保留原来编号i
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit[最好划分的簇类]is: ', bestCentToSplit)
        print('the len of bestClustAss[最好划分簇类中数据的数量] is: ', len(bestClustAss))
        # 将原来簇类中心中的第i类替换
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # replace a centroid with two best centroids
        # 增加第i类产生的新聚类中心
        centList.append(bestNewCents[1, :].tolist()[0])
        # 新簇的分配结果更新,通过找到原来是第i类的数据,直接通过bestClustAss进行替换更新
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss  # reassign new clusters, and SSE
    return mat(centList), clusterAssment


# 测试聚类
def main_test_2():
    dataMat = loadDataSet('testSet.txt')
    dataMat = mat(dataMat)
    biKmeans(dataMat, 4)


import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print(yahooApi)
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())


from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print("error fetching")
        sleep(1)
    fw.close()

def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt


def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p','d', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


if __name__ == '__main__':
    clusterClubs()


