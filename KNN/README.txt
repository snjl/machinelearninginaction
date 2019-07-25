1. kNN.py 一个简单的测试例子
2. kNN_date.py 用1000条约会数据，增加了min-max归一化后的测试例子，参考机器学习第二章
3. createFirstPlot.py 绘图
4. handWriting.py kNN手写体识别

部分代码参考

https://github.com/TrWestdoor/Machine-Learning-in-Action

https://cuijiahua.com/blog/2017/11/ml_1_knn.html

主要参考

https://github.com/pbharrin/machinelearninginaction


# kNN算法的优缺点

### 优点

- 简单好用，容易理解，精度高，理论成熟，既可以用来做分类也可以用来做回归；
- 可用于数值型数据和离散型数据；
- 训练时间复杂度为O(n)；无数据输入假定；
- 对异常值不敏感
### 缺点

- 计算复杂性高；空间复杂性高；
- 样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）；
- 一般数值很大的时候不用这个，计算量太大。但是单个样本又不能太少，否则容易发生误分。
- 最大的缺点是无法给出数据的内在含义。