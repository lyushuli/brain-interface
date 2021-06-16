import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader


def data_concatenate(path1, path2, path3, path_con):
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data3 = pd.read_csv(path3)
    data1['label'] = 0
    data2['label'] = 1
    data3['label'] = 2
    datas = data1
    datas = datas.append(data2, ignore_index=True, sort=False)
    datas = datas.append(data3, ignore_index=True, sort=False)
    datas.to_csv(path_con, index=False)


def get_dataset(path_in, path_train, path_test):
    data = pd.read_csv(path_in, header=None)
    data = data.sample(frac=1)
    trainset_ratio = 0.8
    data_train = data[: int(data.shape[0] * trainset_ratio)]
    data_test = data[int(data.shape[0] * trainset_ratio): ]
    data_train.to_csv(path_train, header=None, index=None)
    data_test.to_csv(path_test, header=None, index=None)


def standard_data(path_in, path_out):
    data_input = pd.read_csv(path_in)
    label = data_input['label']
    data_input = (data_input - data_input.mean()) / data_input.std()
    data_input['label'] = label
    data_input = data_input.drop(['Time'], axis=1)
    data_input.to_csv(path_out, header=False, index=False)


class BrainDataset(Dataset):
    def __init__(self, file_path):
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        x_data = data[:, :-1]
        y_data = data[:, [-1]].flatten()
        y_data.astype(int)
        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.LongTensor(y_data)
        self.len = data.shape[0]

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


def generate_dataloader(path_train, path_test, size=32):
    dataset_train = BrainDataset(path_train)
    dataset_test = BrainDataset(path_test)
    train_data = DataLoader(dataset=dataset_train, batch_size=size, shuffle=False)
    test_data = DataLoader(dataset=dataset_test, batch_size=size, shuffle=False)
    return train_data, test_data


class DeepNN(torch.nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.linear1 = torch.nn.Linear(64, 32)
        self.linear2 = torch.nn.Linear(32, 12)
        self.linear3 = torch.nn.Linear(12, 3)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        return x


def train(epoch):
    for i, (inputs, labels) in enumerate(train_data, 0):
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch=', epoch + 1, ',loss=', loss.item())


def test():
    total_number = 0
    correct_number = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data, 0):
            y_pred = model(inputs)
            _, prediction = torch.max(y_pred, dim=1)
            total_number = total_number + labels.size(0)
            correct_number = correct_number + sum((1 * (prediction == labels))).item()
    accuracy = correct_number / total_number * 100
    print('accuracy=', accuracy, '%')


if __name__ == '__main__':
    file_path0 = './datas/S005R04 T0.csv'
    file_path1 = './datas/S005R04 T1.csv'
    file_path2 = './datas/S005R04 T2.csv'
    file_path = './datas/data.csv'
    file_path_pro = './datas/data_processed.csv'
    file_path_train = './datas/training_data.csv'
    file_path_test = './datas/test_data.csv'
    batch_size = 32
    epochs = 20

    #data_concatenate(file_path0, file_path1, file_path2, file_path)
    #standard_data(file_path, file_path_pro)
    #get_dataset(file_path_pro, file_path_train, file_path_test)

    model = DeepNN()
    # model = ConvNeuralNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    train_data, test_data = generate_dataloader(file_path_train, file_path_test, size=batch_size)

    model.train()
    for i in range(epochs):
        train(i)

    model.eval()
    test()
