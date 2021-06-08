import pandas as pd
import numpy as np
from sklearn import decomposition


def get_dataset():
    kur_path = r"./data/Qiaodu.xlsx"
    kur_data = pd.read_excel(kur_path, sheet_name=0, header=None)
    kur_data = kur_data.to_numpy()

    skew_path = r"./data/Piandu.xlsx"
    skew_data = pd.read_excel(skew_path, sheet_name=0, header=None)
    skew_data = skew_data.to_numpy()

    ave_path = r"./data/Average Bandpower.xlsx"
    ave_data = pd.read_excel(ave_path, sheet_name=0, header=None)
    ave_data = ave_data.to_numpy()

    y = []
    for i in kur_data:
        y.append(int(i[-1]))

    kur_data = kur_data[:, 0:64]
    skew_data = skew_data[:, 0:64]
    ave_data = ave_data[:, 0:64]
    x = np.hstack((kur_data, skew_data, ave_data))
    for i in range(len(x[0])):
        temp = x[:, i]
        temp = (temp - np.min(temp))/ (np.max(temp)-np.min(temp))
        x[:, i] = temp

    pca = decomposition.PCA(n_components=20)
    pca.fit(x, y)
    x_pca = pca.fit_transform(x, y)
    print(pca.explained_variance_)
    return x_pca, y


def data_pre(x, y):
    y = np.array(y)
    y = y.reshape((len(y), 1))
    data = np.hstack((x, y))
    np.random.shuffle(data)

    x = data[:, 0:len(data[0]) - 1]
    y = data[:, len(data[0]) - 1]
    return x, y