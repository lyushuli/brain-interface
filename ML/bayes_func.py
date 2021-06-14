import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from data_process import get_dataset, data_pre, recall_cal, precision_cal, data_2
import numpy as np

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

        clf = GaussianNB(var_smoothing=1e-8)
        clf.fit(x_train, y_train)  # 带入训练集训练模型

        # 预测
        y_test_pre = clf.predict(x_test)  # 利用拟合的贝叶斯进行预测
        acc += sum(y_test_pre == y_test) / num_test  # accuracy
        precis += precision_cal(y_test, y_test_pre)  # precision
        recall += recall_cal(y_test, y_test_pre)  # recall
        f1 += f1_score(y_test, y_test_pre, average="macro")
    print('the accuracy is', acc / test_time)  # 显示预测准确率
    print('the precision is', precis / test_time)
    print('the recall is', recall / test_time)
    print('the f1 is', f1 / test_time)

