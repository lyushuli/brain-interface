import pandas as pd
import numpy as np
from sklearn import decomposition


def get_dataset(type=None):
    kur_path = r"./data/Qiaodu.xlsx"
    kur_data = pd.read_excel(kur_path, sheet_name=0, header=None)
    kur_data = kur_data.to_numpy()

    skew_path = r"./data/Piandu.xlsx"
    skew_data = pd.read_excel(skew_path, sheet_name=0, header=None)
    skew_data = skew_data.to_numpy()

    ave_path = r"./data/Average Bandpower.xlsx"
    ave_data = pd.read_excel(ave_path, sheet_name=0, header=None)
    ave_data = ave_data.to_numpy()

    msf_path = r"./data/mean_square_freq.xlsx"
    msf_data = pd.read_excel(msf_path, sheet_name=0, header=None)
    ms_data = msf_data.to_numpy()

    vf_path = r"./data/v_f.xlsx"
    vf_data = pd.read_excel(vf_path, sheet_name=0, header=None)
    vf_data = vf_data.to_numpy()

    ar_path = r"./data/ar_index.xlsx"
    ar_data = pd.read_excel(ar_path, sheet_name=0, header=None)
    ar_data = ar_data.to_numpy()

    y = []
    for i in kur_data:
        y.append(int(i[-1]))

    kur_data = kur_data[:, 0:64]
    skew_data = skew_data[:, 0:64]
    ave_data = ave_data[:, 0:64]
    ms_data = ms_data[:, 0:64]
    vf_data = vf_data[:, 0:64]

    if type is None or type == 0:
        x = np.hstack((kur_data, skew_data, ave_data, ms_data))
    elif type == 1:
        x = np.hstack((kur_data, skew_data, ave_data, ms_data, vf_data))
    elif type == 2:
        x = np.hstack((kur_data, skew_data, ave_data, ms_data, ar_data))
    else:
        x = np.hstack((kur_data, skew_data, ave_data, ms_data, vf_data, ar_data))

    for i in range(len(x[0])):
        temp = x[:, i]
        temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
        x[:, i] = temp

    pca = decomposition.PCA(n_components=20)
    pca.fit(x, y)
    x_pca = pca.fit_transform(x, y)
    # print(pca.explained_variance_)
    return x_pca, y


def data_pre(x, y):
    y = np.array(y)
    y = y.reshape((len(y), 1))
    data = np.hstack((x, y))
    np.random.shuffle(data)

    x = data[:, 0:len(data[0]) - 1]
    y = data[:, len(data[0]) - 1]
    return x, y


def precision_cal(y_test, y_test_pre):
    precis = np.zeros(3)
    pos = np.argwhere(y_test_pre == 0)
    if len(pos) == 0:
        precis[0] = 0
    else:
        precis[0] = sum(y_test_pre[pos] == y_test[pos] + 0) / len(pos)
    pos = np.argwhere(y_test == 1)
    if len(pos) == 0:
        precis[1] = 0
    else:
        precis[1] = sum(y_test_pre[pos] == y_test[pos] + 0) / len(pos)
    pos = np.argwhere(y_test == 2)
    if len(pos) == 0:
        precis[2] = 0
    else:
        precis[2] = sum(y_test_pre[pos] == y_test[pos] + 0) / len(pos)
    return precis


def recall_cal(y_test, y_test_pre):
    recall = np.zeros(3)
    pos = np.argwhere(y_test == 0)
    if len(pos) == 0:
        recall[0] = 0
    else:
        recall[0] = sum(y_test_pre[pos] == y_test[pos] + 0) / len(pos)
    pos = np.argwhere(y_test == 1)
    if len(pos) == 0:
        recall[1] = 0
    else:
        recall[1] = sum(y_test_pre[pos] == y_test[pos] + 0) / len(pos)
    pos = np.argwhere(y_test == 2)
    if len(pos) == 0:
        recall[2] = 0
    else:
        recall[2] = sum(y_test_pre[pos] == y_test[pos] + 0) / len(pos)
    return recall
