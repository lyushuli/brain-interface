from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_process import get_dataset, data_pre, precision_cal, recall_cal
import numpy as np

"""sklearn中的KNN"""
test_time = 500
for j in range(4):
    print(j)
    x, y = get_dataset(j)
    x, y = data_pre(x, y)
    acc = 0
    precis = np.zeros(3)
    recall = np.zeros(3)
    f1 = 0
    for i in range(test_time):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        num_test = len(y_test)
        # 创建knn类
        knn = KNeighborsClassifier(n_neighbors=7)
        # 训练knn
        knn.fit(x_train, y_train)
        y_test_pre = knn.predict(x_test)
        acc += knn.score(x_test, y_test)
        precis += precision_cal(y_test, y_test_pre)  # precision
        recall += recall_cal(y_test, y_test_pre)  # recall
        f1 += f1_score(y_test, y_test_pre, average="macro")
    print('the accuracy is', acc / test_time)  # 显示预测准确率
    print('the precision is', precis / test_time)
    print('the recall is', recall / test_time)
    print('the f1 is', f1 / test_time)
