import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from data_process import get_dataset, data_pre
import numpy as np

x, y = get_dataset()
x, y = data_pre(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
num_test = len(y_test)

clf = GaussianNB(var_smoothing=1e-8)
clf.fit(x_train, y_train)  # 带入训练集训练模型

# 预测
y_test_pre = clf.predict(x_test)  # 利用拟合的贝叶斯进行预测
print('the predict values are', y_test_pre)  # 显示结果
acc = sum(y_test_pre == y_test) / num_test
print('the accuracy is', acc)  # 显示预测准确率

