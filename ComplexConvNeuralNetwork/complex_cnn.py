import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from data_label import generate_dataloader
from sklearn import metrics
from matplotlib import pyplot as plt


class InceptionA(torch.nn.Module):
    def __init__(self, in_channel):
        super(InceptionA, self).__init__()
        self.branch_pool = torch.nn.Conv2d(in_channel, 24, kernel_size=1)

        self.branch1x1 = torch.nn.Conv2d(in_channel, 16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channel, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channel, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.pool = torch.nn.AvgPool2d(kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        branch_pool = self.pool(x)
        branch_pool = self.branch_pool(branch_pool)

        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        outputs = [branch_pool, branch1x1, branch5x5, branch3x3]
        return torch.cat(outputs, dim=1)


class ComplexConvNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(ComplexConvNeuralNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1)
        self.incep1 = InceptionA(in_channel=10)
        self.conv2 = torch.nn.Conv2d(in_channels=88, out_channels=20, kernel_size=3, padding=1)
        self.incep2 = InceptionA(in_channel=20)
        self.pool = torch.nn.MaxPool2d(2)
        self.linear1 = torch.nn.Linear(in_features=88 * 4 * 4, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=3)
        self.conv3 = torch.nn.Conv2d(in_channels=88, out_channels=88, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.view(batch_size, 1, 8, 8)
        x = F.relu(self.conv1(x))
        x = self.incep1(x)
        x1 = x
        x = F.relu(self.conv2(x))
        x = self.incep2(x)
        x = F.relu(x + x1)
        x = self.pool(self.conv3(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    '''0.92
    def forward(self, x):
        x = x.view(batch_size, 1, 8, 8)
        x = F.relu(self.conv1(x))
        x = self.incep1(x)
        x1 = x
        x = F.relu(self.conv2(x))
        x = self.incep2(x)
        x = F.relu(x + x1)
        x = self.pool(x)
        x = x.view(batch_size, -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    '''
    ''' 0.89
    def forward(self, x):
        x = x.view(batch_size, 1, 8, 8)
        x = F.relu(self.conv1(x))
        x = self.incep1(x)
        x = F.relu(self.pool(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(batch_size, -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    '''


def train(epoch):
    loss_total = 0
    for i, (inputs, labels) in enumerate(train_data, 0):
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total = loss_total + loss.item()
    print('epoch=', epoch + 1, ',loss=', loss_total / batch_size)
    writer.add_scalar('loss', loss_total / batch_size, epoch + 1)
    return loss_total/batch_size


def test(epoch):
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
    writer.add_scalar('accuracy', accuracy, epoch + 1)

    if epoch == epochs - 1:
        mat = confusion_matrix()
        precision_value = precision(mat)
        recall_value = recall(mat)
        show_result(precision_value, recall_value, mat)
    return accuracy


def confusion_matrix():
    con_mat = {'T0': {'T0': 0, 'T1': 0, 'T2': 0},
               'T1': {'T0': 0, 'T1': 0, 'T2': 0},
               'T2': {'T0': 0, 'T1': 0, 'T2': 0}}
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data, 0):
            y_pred = model(inputs)
            _, prediction = torch.max(y_pred, dim=1)
            for j in range (0, batch_size):
                actual = classes[labels[j].item()]
                predict = classes[prediction[j].item()]
                con_mat[actual][predict] = con_mat[actual][predict] + 1
    return con_mat


def precision(matrix):
    precision_dict = {'T0': 0, 'T1': 0, 'T2': 0}
    for i in classes:
        true_predict_num = matrix[i][i]
        all_predict_num = 0
        for j in classes:
            all_predict_num = all_predict_num + matrix[j][i]
        precision_dict[i] = round(true_predict_num / all_predict_num, 4)
    return precision_dict


def recall(matrix):
    recall_dict = {'T0': 0, 'T1': 0, 'T2': 0}
    for i in classes:
        true_predict_num = matrix[i][i]
        all_predict_num = 0
        for j in classes:
            all_predict_num += matrix[i][j]
        recall_dict[i] = round(true_predict_num / all_predict_num, 4)
    return recall_dict


def f1_score(p, f):
    return round(2 * p * f / (p + f), 2)


def accuracy(classes, matrix):
    right_num = 0
    sum_num = 0
    for row in classes:
        right_num += matrix[row][row]
        for column in classes:
            sum_num += matrix[row][column]
    return round(right_num / sum_num, 4)


def show_result(precision, recall, matrix):
    print('\t\t', 'precision', '\t', 'recall', '\t', 'F1_score')
    for item in classes:
        print(item, '\t', precision[item], '\t', recall[item], '\t', f1_score(precision[item], recall[item]))
    print(matrix)


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
    classes = ['T0', 'T1', 'T2']

    model = ComplexConvNeuralNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    writer = SummaryWriter('./data_record/model')
    dummy_input = torch.rand(batch_size, 64)
    writer.add_graph(model, (dummy_input,))

    train_data, test_data = generate_dataloader(file_path_train, file_path_test, size=batch_size)

    loss_list = []
    acc_list = []
    model.train()
    for i in range(epochs):
        model.train()
        loss_list.append(train(i))
        model.eval()
        acc_list.append(test(i))

    plt.figure()
    plt.plot(loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    plt.figure()
    plt.plot(acc_list)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()
    writer.close()