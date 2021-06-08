import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_process import get_dataset, data_pre
import numpy as np

"""sklearn中的KNN"""
x, y = get_dataset()
x, y = data_pre(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
num_test = len(y_test)
# 创建knn类
knn = KNeighborsClassifier()
# 训练knn
knn.fit(x_train, y_train)
y_test_pre = knn.predict(x_test)
print('the predict values are', y_test_pre)  # 显示结果
acc = sum(y_test_pre == y_test) / num_test
print('the accuracy is', acc)  # 显示预测准确率
print(knn.score(x_test, y_test))
