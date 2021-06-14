import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from data_process import get_dataset, data_pre, data_2
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


if __name__ == "__main__":
    for j in range(4):
        x, y = get_dataset(j)
        x, y = data_pre(x, y)
        x, y = data_2(x, y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        num_test = len(y_test)
        clf = KNeighborsClassifier(n_neighbors=4)
        # 训练knn
        clf.fit(x_train, y_train)
        y_test_pre = clf.predict(x_test)  # 利用拟合的贝叶斯进行预测
        acc = sum(y_test_pre == y_test) / num_test  # accuracy
        con_mat = confusion_matrix(y_test, y_test_pre)
        con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
        con_mat_norm = np.around(con_mat_norm, decimals=2)
        f1 = f1_score(y_test, y_test_pre)
        plt.figure()
        sns.heatmap(con_mat_norm, annot=True, cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()
        print(j)
        print('the accuracy is', acc)  # 显示预测准确率
        print('the f1_score is', f1)