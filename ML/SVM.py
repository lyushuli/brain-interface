import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, decomposition
from sklearn.datasets import make_blobs
from sklearn.metrics import recall_score, f1_score
from mpl_toolkits.mplot3d import Axes3D
from data_process import get_dataset, data_pre, precision_cal, recall_cal
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


def SVM_view(x, y):
    pca = decomposition.PCA(n_components=2)
    pca.fit(x, y)
    x = pca.fit_transform(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(x_train, y_train)
    y_test_pre = clf.predict(x_test)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap="rainbow")
    plt.xticks([])  # 隐藏X轴刻度
    plt.yticks([])  # 隐藏Y轴刻度
    # 获取平面上两条坐标轴的最大值和最小值
    ax = plt.gca()  # 获取当前的子图，如果不存在，则创建新的子图
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # 在最大值和最小值之间形成30个规律的数据
    axisx = np.linspace(xlim[0], xlim[1], 30)
    axisy = np.linspace(ylim[0], ylim[1], 30)
    axisx, axisy = np.meshgrid(axisx, axisy)
    # 我们将使用这里形成的二维数组作为我们contour函数中的X和Y
    # 使用meshgrid函数将两个一维向量转换为特征矩阵
    # 核心是将两个特征向量广播，以便获取y.shape * x.shape这么多个坐标点的横坐标和纵坐标
    xy = np.vstack([axisx.ravel(), axisy.ravel()]).T
    plt.scatter(xy[:, 0], xy[:, 1], s=1)
    Z = clf.decision_function(xy).reshape(axisx.shape)
    ax.contour(axisx, axisy, Z
               , colors="k"
               , levels=[-1, 0, 1]  # 画三条等高线，分别是Z为-1，Z为0和Z为1的三条线
               , alpha=0.5
               , linestyles=["--", "-", "--"])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()

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


SVM_view(x, y2)
x_train0, x_test0, y_train0, y_test0 = train_test_split(x, y0, test_size=0.3)
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y1, test_size=0.3)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y2, test_size=0.3)
num_test = len(y_test0)

# fit
clf = svm.SVC()  # 默认为rbf
clf.fit(x_train0, y_train0)
y_test_pre = clf.predict(x_test0)
print('the predict 0 are', y_test_pre)  # 显示结果
acc = sum(y_test_pre == y_test0) / num_test
precis = precision_cal(y_test0, y_test_pre)  # precision
recall = recall_cal(y_test0, y_test_pre)  # recall
f1 = f1_score(y_test0, y_test_pre, average="macro")
print('the 0 accuracy is', acc)  # 显示预测准确率
print('the 0 precision is', precis)
print('the 0 recall is', recall)
print('the 0 f1 is', f1)

clf = svm.SVC()
clf.fit(x_train1, y_train1)
y_test_pre = clf.predict(x_test1)
print('the predict 1 are', y_test_pre)  # 显示结果
acc = sum(y_test_pre == y_test1) / num_test
precis = precision_cal(y_test1, y_test_pre)  # precision
recall = recall_cal(y_test1, y_test_pre)  # recall
f1 = f1_score(y_test1, y_test_pre, average="macro")
print('the 1 accuracy is', acc)  # 显示预测准确率
print('the 1 precision is', precis)
print('the 1 recall is', recall)
print('the 1 f1 is', f1)

clf = svm.SVC()
clf.fit(x_train2, y_train2)
y_test_pre = clf.predict(x_test2)
print('the predict 2 are', y_test_pre)  # 显示结果
acc = sum(y_test_pre == y_test2) / num_test
precis = precision_cal(y_test2, y_test_pre)  # precision
recall = recall_cal(y_test2, y_test_pre)  # recall
f1 = f1_score(y_test2, y_test_pre, average="macro")
print('the 2 accuracy is', acc)  # 显示预测准确率
print('the 2 precision is', precis)
print('the 2 recall is', recall)
print('the 2 f1 is', f1)

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
        # 一对多分类
        num_test = len(y_test)
        clf = OneVsRestClassifier(svm.SVC())
        clf.fit(x_train, y_train)
        y_test_pre = clf.predict(x_test)
        acc += sum(y_test_pre == y_test) / num_test
        precis += precision_cal(y_test, y_test_pre)  # precision
        recall += recall_cal(y_test, y_test_pre)  # recall
        f1 += f1_score(y_test, y_test_pre, average="macro")
    print('the accuracy is', acc/test_time)  # 显示预测准确率
    print('the precision is', precis/test_time)
    print('the recall is', recall/test_time)
    print('the f1 is', f1/test_time)
