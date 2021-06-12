from sklearn import datasets, tree
import numpy as np
from data_process import get_dataset, data_pre
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


x, y = get_dataset()
x, y = data_pre(x, y)

test_time = 500
acc = 0
for i in range(test_time):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    num_test = len(y_test)
    # 训练
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    # 预测
    y_test_pre = clf.predict(x_test)
    acc += sum(y_test_pre == y_test) / num_test
print('the accuracy is', acc/test_time)  # 显示预测准确率




