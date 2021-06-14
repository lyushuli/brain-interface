from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from data_process import get_dataset_nopca, data_pre, precision_cal, recall_cal
import numpy as np

test_time = 10
for j in range(4):
    print(j)
    x, y = get_dataset_nopca(j)
    x, y = data_pre(x, y)
    acc = 0
    precis = np.zeros(3)
    recall = np.zeros(3)
    f1 = 0
    for i in range(test_time):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        num_test = len(y_test)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 200, 20), max_iter=10000000, )
        clf.fit(x_train, y_train)
        y_test_pre = clf.predict(x_test)
        acc += sum(y_test_pre == y_test) / num_test
        precis += precision_cal(y_test, y_test_pre)  # precision
        recall += recall_cal(y_test, y_test_pre)  # recall
        f1 += f1_score(y_test, y_test_pre, average="macro")
    print('the accuracy is', acc / test_time)  # 显示预测准确率
    print('the precision is', precis / test_time)
    print('the recall is', recall / test_time)
    print('the f1 is', f1 / test_time)