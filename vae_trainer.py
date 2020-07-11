import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.utils import save_image
import time
import torchvision
from dataset import *


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.latent_dim = 10
        self.mid_layer = None  # 记录中间层的值

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.AvgPool2d(kernel_size=15, stride=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.en_mu = nn.Linear(32 * 7 * 7, self.latent_dim)
        self.en_log_var = nn.Linear(32 * 7 * 7, self.latent_dim)
        self.linear_to_cnn = nn.Linear(self.latent_dim, 32 * 7 * 7)
        self.decoder = nn.Sequential(  # input 32,7,7
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        # x = x.view(-1, 1, 28, 28)
        result = self.encoder(x)
        h = torch.flatten(result, start_dim=1)
        return self.en_mu(h), self.en_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        self.mid_layer = z.detach().cpu().numpy()
        # list1.append(z.cpu().numpy())
        input_ = (self.linear_to_cnn(z)).view(-1, 32, 7, 7)
        result = self.decoder(input_)
        result = result.view(-1, 1, 28, 28)
        # result = self.final_layer(result)
        return result

    def forward(self, x, is_latent=False):
        if not is_latent:
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
        else:
            z = x
            mu = None
            log_var = None
        generation_images = self.decode(z)
        return generation_images, mu, log_var


class VAE_FC(nn.Module):
    def __init__(self, image_size=784):
        super(VAE_FC, self).__init__()
        self.latent_dim = 10
        self.mid_layer = None  # 记录中间层的值
        self.en_hidden = nn.Linear(image_size, 196)
        self.en_mu = nn.Linear(196, self.latent_dim)
        self.en_log_var = nn.Linear(196, self.latent_dim)
        self.de_hidden = nn.Linear(self.latent_dim, 196)
        self.de_out = nn.Linear(196, image_size)

    def encode(self, x):
        x = x.view(-1, 784)
        relu = nn.ReLU()
        h = relu(self.en_hidden(x))
        return self.en_mu(h), self.en_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        self.mid_layer = z.detach().cpu().numpy()
        relu = nn.ReLU()
        h = relu(self.de_hidden(z))
        return torch.sigmoid(self.de_out(h)).view(-1, 1, 28, 28)

    def forward(self, x, is_latent=False):
        if not is_latent:
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
        else:
            z = x
            mu = None
            log_var = None
        generation_images = self.decode(z)
        return generation_images, mu, log_var


class VAETrainer:
    def __init__(self, train_img, auto_encoder, channel_num=1, rows=28, cols=28):
        self.trainImg = torch.tensor(train_img, dtype=torch.float32)
        self.autoEncoder = auto_encoder
        self.channelNum = channel_num
        self.rows = rows
        self.cols = cols

    def training(self, epoch_nums=100, batch_size=128, learning_rate=0.002, output_dir="vae_images",
                 state_dict_output_path=None):
        os.makedirs(output_dir, exist_ok=True)  # create file folder
        print("Output images will be saved in \"{}\".".format(os.path.abspath(output_dir)))

        # If CUDA is valid, we use CUDA for deep learning. Or we use CPU.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("We use {} for training.".format(device))
        self.autoEncoder = self.autoEncoder.to(device)

        # self.autoEncoder.load_state_dict(torch.load("mnist_vae_state.pth"))
        optimizer = optim.Adam(self.autoEncoder.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        loss_func = nn.BCELoss(reduction='sum')  # MSE is also OK.

        for epoch in range(epoch_nums):
            print("epoch ", epoch + 1, '/', epoch_nums, sep='')
            t1 = time.process_time()
            batch_num = self.trainImg.shape[0] // batch_size
            for batch_index in range(batch_num):
                origin_images = self.trainImg[batch_index * batch_size:(batch_index + 1) * batch_size].clone().to(
                    device)
                generation_images, mu, log_var = self.autoEncoder(origin_images)

                encoder_decoder_loss = loss_func(generation_images, origin_images)
                kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = encoder_decoder_loss + kl_div

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # When the epoch is finished, print result and save origin and output images.
                if batch_index + 1 == batch_num:
                    print("loss: {:.3f} + {:.3f} = {:.3f} Time: {:.2f}s".format(encoder_decoder_loss.item(),
                                                                                kl_div.item(), loss.item(),
                                                                                time.process_time() - t1))
                    with torch.no_grad():
                        origin_example = origin_images[:50].detach()
                        origin_example = origin_example.view(origin_example.shape[0],
                                                             self.channelNum, self.rows, self.cols)
                        decoder_example = generation_images[:50].detach()
                        decoder_example = decoder_example.view(decoder_example.shape[0],
                                                               self.channelNum, self.rows, self.cols)
                        output_example = torch.cat((origin_example, decoder_example), dim=0)
                        save_image(output_example,
                                   os.path.join(output_dir, "epoch{}.png".format(epoch + 1)),
                                   nrow=10)
        print("Finish training. You can see output in \"{}\".".format(os.path.abspath(output_dir)))
        if state_dict_output_path:
            torch.save(self.autoEncoder.state_dict(), state_dict_output_path)


class VAEAnalyzer:
    def __init__(self, test_img, test_label, auto_encoder, state_dict_path):
        self.testImg = torch.tensor(test_img, dtype=torch.float32)
        self.testLabel = test_label
        self.autoEncoder = auto_encoder
        self.state_dict_path = state_dict_path
        self.autoEncoder.load_state_dict(torch.load(self.state_dict_path, map_location=torch.device('cpu')))

    def analysis(self):
        # We don't train the network. So CPU is enough. CUDA may have the problem with the lack of video memory.
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device('cpu')
        # print("We use {} for training.".format(device))
        self.autoEncoder = self.autoEncoder.to(device)
        self.testImg = self.testImg.to(device)
        # self.testLabel = self.testLabel.to(device)

        with torch.no_grad():
            self.autoEncoder(self.testImg)
            mid_layer = self.autoEncoder.mid_layer
            self.t_sne(mid_layer, self.testLabel, "Latent Vectors of FC-VAE")

            # d = dict()
            # for i in range(10):
            #     d[i] = mid_layer[self.testLabel == i]

            # from sklearn import mixture
            # clf = mixture.GaussianMixture(n_components=10, covariance_type='full')
            # for i in range(10):
            #     clf.fit(d[i])  # 对单个数字求概率分布
            #     sam, _ = clf.sample(100)
            #     output_images, _, _ = self.autoEncoder(torch.Tensor(sam), is_latent=True)
            #     save_image(output_images, "sample{}.jpg".format(i), nrow=10)
            # clf = mixture.GaussianMixture(n_components=20, covariance_type='full')
            # clf.fit(mid_layer)  # 不分类别 对所有数据求概率分布
            # sam, _ = clf.sample(100)
            # output_images, _, _ = self.autoEncoder(torch.Tensor(sam), is_latent=True)
            # save_image(output_images, "sample.jpg", nrow=10)

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
            # plt.savefig(fname="graphs/{}.pdf".format(title.replace(' ', '_')), format="pdf")
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
    # ds = ManualDataset(True, "MNIST", "MNIST")
    ds = TorchReaderDataset('./data')

    state_dict = "mnist_vae_state_fc.pth"
    output_dir = "vae_images_fc_10"
    vae_network = VAE_FC()

    vae_trainer = VAETrainer(ds.get_matrix_train_img_with_channel(True), vae_network)
    vae_trainer.training(epoch_nums=30, batch_size=256, learning_rate=0.005,
                         output_dir=output_dir, state_dict_output_path=state_dict)
    vae_analyzer = VAEAnalyzer(ds.get_matrix_test_img_with_channel(True), ds.get_vector_test_label(),
                               vae_network, state_dict_path=state_dict)
    vae_analyzer.analysis()
