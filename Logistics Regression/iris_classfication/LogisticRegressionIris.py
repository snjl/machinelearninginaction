from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# def loadDataSet():
data = pd.read_csv('iris.csv')
data = data.sample(frac=1)
y_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
data['species'] = data['species'].map(y_map)
y = data['species']
data.drop('species', axis=1, inplace=True)

# 也可以不转化为numpy形式，也可以计算
# y = y.to_numpy()
# data = data.to_numpy()

# 增加min-max归一化
# min_max_scaler = preprocessing.MinMaxScaler()
# data = min_max_scaler.fit_transform(data)

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)

classifier = LogisticRegression(max_iter=55).fit(x_train, y_train)
predict_test_data = classifier.predict(x_test)
print(predict_test_data)
print(y_test)

print('accuracy:', classifier.score(x_test, y_test))
