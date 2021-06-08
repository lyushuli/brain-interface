import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from data_process import get_dataset, data_pre
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

x, y = get_dataset()
x, y = data_pre(x, y)

y0 = []
for i in y:
    if i == 2 or i == 1:
        y0.append(0)
    else:
        y0.append(1)
y0 = np.array(y0)

y1 = []
for i in y:
    if i == 0 or i == 2:
        y1.append(0)
    else:
        y1.append(1)
y1 = np.array(y1)

y2 = []
for i in y:
    if i == 0 or i == 1:
        y2.append(0)
    else:
        y2.append(1)
y2 = np.array(y2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_train0, x_test0, y_train0, y_test0 = train_test_split(x, y0, test_size=0.3)
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y1, test_size=0.3)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y2, test_size=0.3)

# 一对多分类
num_test = len(y_test1)
print(y_test)
clf = OneVsRestClassifier(svm.SVC())
clf.fit(x_train, y_train)
y_test_pre = clf.predict(x_test)
print('the predict are', y_test_pre)  # 显示结果
acc = sum(y_test_pre == y_test) / num_test
print('the accuracy is', acc)  # 显示预测准确率


# fit
clf = svm.SVC()  # 默认为rbf
clf.fit(x_train0, y_train0)
y_test_pre = clf.predict(x_test0)
print('the predict 0 are', y_test_pre)  # 显示结果
acc = sum(y_test_pre == y_test0) / num_test
print('the 0 accuracy is', acc)  # 显示预测准确率

clf = svm.SVC()
clf.fit(x_train1, y_train1)
y_test_pre = clf.predict(x_test1)
print('the predict 1 are', y_test_pre)  # 显示结果
acc = sum(y_test_pre == y_test1) / num_test
print('the 1 accuracy is', acc)  # 显示预测准确率

clf = svm.SVC()
clf.fit(x_train2, y_train2)
y_test_pre = clf.predict(x_test2)
print('the predict 2 are', y_test_pre)  # 显示结果
acc = sum(y_test_pre == y_test2) / num_test
print('the 2 accuracy is', acc)  # 显示预测准确率


