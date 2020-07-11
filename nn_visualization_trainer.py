from dataset import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time


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
        self.fcSeq1 = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(),
        )
        self.fcSeq2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.fcSeq3 = nn.Linear(84, 10)

        # visualization
        self.mid_layer = None

    def forward(self, x):
        tmp = self.convSeq(x)
        tmp = tmp.view(x.shape[0], -1)
        tmp = self.fcSeq1(tmp)
        tmp = self.fcSeq2(tmp)
        self.mid_layer = tmp
        output = self.fcSeq3(tmp)
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
        torch.save(self.net.state_dict(), "test.pth")

    def accuracy(self):
        with torch.no_grad():
            outputs = self.net(self.testImg.to(self.device))
            _, prediction = torch.max(outputs, 1)
            correct = (prediction == self.testLabel.to(self.device)).sum().item()
            return correct / prediction.shape[0]


class NNAnalyzer:
    def __init__(self, test_img, test_label, net, state_dict_path="test.pth"):
        self.testImg = torch.tensor(test_img, dtype=torch.float32)
        self.testLabel = torch.tensor(test_label)
        self.net = net
        self.state_dict_path = state_dict_path
        self.net.load_state_dict(torch.load(self.state_dict_path, map_location=torch.device('cpu')))

    def analysis(self):
        # We don't train the network. So CPU is enough. CUDA may have the problem with the lack of video memory.
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device('cpu')
        print("We use {} for training.".format(device))
        self.net = self.net.to(device)
        self.testImg = self.testImg.to(device)
        # self.testLabel = self.testLabel.to(device)

        with torch.no_grad():
            self.net(self.testImg)
            mid_layer = self.net.mid_layer
            self.t_sne(mid_layer.detach().cpu().numpy(), self.testLabel.detach().cpu().numpy())

    def t_sne(self, visual_part, labels=None, title=None):
        from sklearn import manifold
        import matplotlib.pyplot as plt

        def plot_embedding(X, y=None, title=None):
            x_min, x_max = np.min(X, 0), np.max(X, 0)
            X = (X - x_min) / (x_max - x_min)

            # plt.figure()
            # ax = plt.subplot(111)
            if y is None:
                plt.scatter(X[:, 0], X[:, 1], marker='.')
            else:
                for i in range(X.shape[0]):
                    plt.text(X[i, 0], X[i, 1], str(y[i]),
                             color=plt.cm.Set1(y[i] / 10.),
                             fontdict={'weight': 'bold', 'size': 9})

            plt.xticks([]), plt.yticks([])
            if title is not None:
                plt.title(title)
            # plt.savefig(fname="fc.pdf", format="pdf")
            plt.show()

        print("Do T-SNE...")
        tsne = manifold.TSNE(n_components=2)  # , init='random', random_state=None)
        X_tsne = tsne.fit_transform(visual_part[:1000])
        print("Plotting...")
        Y = None
        if labels is not None:
            Y = labels[:1000]
        plot_embedding(X_tsne, Y, title)

if __name__ == '__main__':
    # Choose cuda or cpu automatically.
    # ds = ManualDataset(True, "MNIST", "MNIST")
    ds = TorchReaderDataset('./data')
    nn_trainer = NNTrainer(ds.get_matrix_train_img_with_channel(is_normalized=True), ds.get_vector_train_label(),
                           ds.get_matrix_test_img_with_channel(is_normalized=True), ds.get_vector_test_label(),
                           LeNetReLU())  # Accuracy: 0.9809 Time: 542.734375
    nn_trainer.mini_batch_training(batch_size=5000, learning_rate=0.05, epoch_nums=20)
    nn_analyzer = NNAnalyzer(ds.get_matrix_test_img_with_channel(is_normalized=True),
                             ds.get_vector_test_label(),
                             LeNetReLU())
    nn_analyzer.analysis()
