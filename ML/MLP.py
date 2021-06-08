from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from data_process import get_dataset, data_pre
import numpy as np

x, y = get_dataset()
x, y = data_pre(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
num_test = len(y_test)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(600, 400, 200, 20), max_iter=1000000000, )
clf.fit(x_train, y_train)
y_test_pre = clf.predict(x_test)
print('the predict values are', y_test_pre)  # 显示结果
acc = sum(y_test_pre == y_test) / num_test
print('the accuracy is', acc)  # 显示预测准确率
print(clf.score(x_test, y_test))
