from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from data_process import get_dataset, data_pre


x, y = get_dataset()
x, y = data_pre(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
num_test = len(y_test)
# 构建决策树
clf = tree.DecisionTreeClassifier()  # 建立决策树对象
clf.fit(x_train, y_train)  # 决策树拟合
tree.plot_tree(clf)
plt.show()
# 预测
y_test_pre = clf.predict(x_test)
print('the predict values are', y_test_pre)  # 显示结果

acc = sum(y_test_pre == y_test) / num_test
print('the accuracy is', acc)  # 显示预测准确率