from dataset import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(784, 196)
        self.relu = nn.ReLU()
        self.output = nn.Linear(196, 10)

    def forward(self, x):
        tmp = self.hidden(x)
        tmp = self.relu(tmp)
        output = self.output(tmp)
        return output


class LeNetReLU(nn.Module):
    def __init__(self):
        super(LeNetReLU, self).__init__()
        self.convSeq = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fcSeq = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        tmp = self.convSeq(x)
        tmp = tmp.view(x.shape[0], -1)
        output = self.fcSeq(tmp)
        return output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convSeq = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.fcSeq = nn.Sequential(
            nn.Linear(32*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        tmp = self.convSeq(x)
        tmp = tmp.view(x.shape[0], -1)
        output = self.fcSeq(tmp)
        return output


class LeNetSigmoid(nn.Module):  # Useless
    def __init__(self):
        super(LeNetSigmoid, self).__init__()
        self.convSeq = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fcSeq = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        tmp = self.convSeq(x)
        tmp = tmp.view(x.shape[0], -1)
        output = self.fcSeq(tmp)
        return output


class NNTrainer:
    def __init__(self, train_img, train_label, test_img, test_label, net):
        self.trainImg = torch.tensor(train_img, dtype=torch.float32)
        self.trainLabel = torch.tensor(train_label)
        self.testImg = torch.tensor(test_img, dtype=torch.float32)
        self.testLabel = torch.tensor(test_label)
        self.net = net
        self.device = None
        # self.check_cuda()

    def check_cuda(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Now use device:", self.device)
        self.net = self.net.to(self.device)
        # self.trainImg = self.trainImg.to(self.device)
        # self.trainLabel = self.trainLabel.to(self.device)
        # self.testImg = self.testImg.to(self.device)
        # self.testLabel = self.testLabel.to(self.device)

    def mini_batch_training(self, batch_size=5000, learning_rate=0.001, epoch_nums=20):
        self.check_cuda()
        # print("Now use device:", self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)

        t1 = time.process_time()
        for epoch in range(epoch_nums):
            print("epoch ", epoch + 1, '/', epoch_nums, sep='')
            running_loss = 0.0
            for batch_index in range(self.trainImg.shape[0] // batch_size):
                inputs = self.trainImg[batch_index * batch_size:(batch_index+1) * batch_size].to(self.device)
                labels = self.trainLabel[batch_index * batch_size:(batch_index+1) * batch_size].to(self.device)
                # print(inputs.shape, labels.shape, inputs.dtype, labels.dtype)
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print("Loss: {0:.4f}".format(running_loss), "Accuracy:", self.accuracy())
        t2 = time.process_time()
        print("Finish training.")
        print("Accuracy:", self.accuracy())
        print("Time:", t2 - t1)
        # torch.save(self.net.state_dict(), "mnist_cnn_state.pth")

    def accuracy(self):
        with torch.no_grad():
            outputs = self.net(self.testImg.to(self.device))
            _, prediction = torch.max(outputs, 1)
            correct = (prediction == self.testLabel.to(self.device)).sum().item()
            return correct / prediction.shape[0]


if __name__ == '__main__':
    # Choose cuda or cpu automatically.
    # ds = ManualDataset(True, "MNIST", "MNIST")
    ds = TorchReaderDataset('./data')
    nn_trainer = NNTrainer(ds.get_vector_train_img(is_normalized=True), ds.get_vector_train_label(),
                           ds.get_vector_test_img(is_normalized=True), ds.get_vector_test_label(),
                           MLP())  # 0.9618 0.9623 6.4375 46.203125
    # nn_trainer = NNTrainer(ds.get_matrix_train_img_with_channel(is_normalized=True), ds.get_vector_train_label(),
    #                        ds.get_matrix_test_img_with_channel(is_normalized=True), ds.get_vector_test_label(),
    #                        LeNetReLU())  # Accuracy: 0.978 0.9843 Time: 69.25 625.796875
    # nn_trainer = NNTrainer(ds.get_matrix_train_img_with_channel(is_normalized=True), ds.get_vector_train_label(),
    #                        ds.get_matrix_test_img_with_channel(is_normalized=True), ds.get_vector_test_label(),
    #                        CNN())  # Accuracy: 0.9905 Time: 255.015625 GPU
    nn_trainer.mini_batch_training(batch_size=3000, learning_rate=0.05, epoch_nums=20)
