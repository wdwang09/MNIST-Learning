from dataset import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.utils import save_image  # Use this function to save the fake images
# from torch.autograd import Variable
import time


class Generator(nn.Module):
    def __init__(self, channel_num, rows, cols):
        super(Generator, self).__init__()
        self.channel_num = channel_num
        self.rows = rows
        self.cols = cols
        self.latent_dim = 100

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, self.channel_num * self.rows * self.cols),  # pixel vector
            nn.Tanh()
        )

    def forward(self, z):
        tmp = self.model(z)
        output_img = tmp.view(tmp.size(0), self.channel_num, self.rows, self.cols)
        return output_img


class Discriminator(nn.Module):
    def __init__(self, channel_num, rows, cols):
        super(Discriminator, self).__init__()
        self.channel_num = channel_num
        self.rows = rows
        self.cols = cols

        self.model = nn.Sequential(
            nn.Linear(self.channel_num * self.rows * self.cols, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.model(img.view(img.shape[0], -1))


class MLPDiscriminator(nn.Module):
    def __init__(self, channel_num, rows, cols):
        super(MLPDiscriminator, self).__init__()
        self.channel_num = channel_num
        self.rows = rows
        self.cols = cols
        self.hidden = nn.Linear(self.channel_num * self.rows * self.cols, 196)
        self.relu = nn.ReLU()
        self.output = nn.Linear(196, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tmp = self.hidden(x.view(x.shape[0], -1))
        tmp = self.relu(tmp)
        output = self.sigmoid(self.output(tmp))
        return output


class GANTrainer:
    def __init__(self, train_img, generator, discriminator, latent_dim=100):
        self.trainImg = torch.tensor(train_img, dtype=torch.float32)
        self.generator = generator
        self.discriminator = discriminator
        self.latentDimension = latent_dim  # same as the input of generator

    def training(self, epoch_nums=100, batch_size=128, learning_rate=0.002, where_to_store_fake_images="images"):
        os.makedirs(where_to_store_fake_images, exist_ok=True)  # create file folder
        print("Fake images will be saved in \"{}\".".format(os.path.abspath(where_to_store_fake_images)))

        # If CUDA is valid, we use CUDA for deep learning. Or we use CPU.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("We use {} for training.".format(device))

        adversarial_loss = nn.BCELoss().to(device)  # loss function

        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        
        # different learning_rate or betas => different result
        optimizer_G = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        for epoch in range(epoch_nums):
            print("epoch ", epoch + 1, '/', epoch_nums, sep='')
            t1 = time.process_time()
            batch_num = self.trainImg.shape[0] // batch_size
            for batch_index in range(batch_num):
                real_images = self.trainImg[batch_index * batch_size:(batch_index+1) * batch_size].to(device)
                curr_batch_size = real_images.shape[0]

                # true_labels = Variable(Tensor(curr_batch_size, 1).fill_(1.0), requires_grad=False)
                # false_labels = Variable(Tensor(curr_batch_size, 1).fill_(0.0), requires_grad=False)
                # true_labels = Tensor(curr_batch_size, 1).fill_(1.0)
                # false_labels = Tensor(curr_batch_size, 1).fill_(0.0)
                true_labels = torch.ones((curr_batch_size, 1)).to(device)
                false_labels = torch.zeros((curr_batch_size, 1)).to(device)
                # real_images = Variable(curr_batch_images.type(Tensor))
                # real_images = curr_batch_images.type(Tensor)

                # generator
                optimizer_G.zero_grad()
                # z = Variable(Tensor(np.random.normal(0, 1, (curr_batch_size, self.latentDimension))))  # random noise
                # z = Tensor(np.random.normal(0, 1, (curr_batch_size, self.latentDimension)))  # random input noise
                z = torch.tensor(np.random.normal(0, 1, (curr_batch_size, self.latentDimension)),
                                 dtype=torch.float32).to(device)
                fake_images = self.generator(z)  # the numbers of fake images is same as z.shape[0]
                g_loss = adversarial_loss(self.discriminator(fake_images), true_labels)
                g_loss.backward()
                optimizer_G.step()

                # discriminator
                optimizer_D.zero_grad()
                real_loss = adversarial_loss(self.discriminator(real_images), true_labels)
                fake_loss = adversarial_loss(self.discriminator(fake_images.detach()), false_labels)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

                # When the epoch is finished, print result and save fake images.
                if batch_index + 1 == batch_num:
                    print("D_loss: {:.4f} G_loss: {:.4f} Time: {:.2f}s".format(
                        d_loss.item(), g_loss.item(), time.process_time()-t1))
                    # only show at most 100 fake images
                    save_image(fake_images.data[:100],
                               os.path.join(where_to_store_fake_images, "epoch{}.png".format(epoch+1)),
                               nrow=10)
        print("Finish training. You can see fake images in \"{}\".".format(os.path.abspath(where_to_store_fake_images)))


if __name__ == '__main__':
    # The architecture is similar to
    # https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/gan/gan.py
    # ds = ManualDataset(True, "MNIST", "MNIST")
    ds = TorchReaderDataset('./data')
    gan_trainer = GANTrainer(ds.get_matrix_train_img_with_channel(is_normalized=True),
                             Generator(1, 28, 28), Discriminator(1, 28, 28))
    gan_trainer.training(epoch_nums=100, batch_size=128, learning_rate=0.002, where_to_store_fake_images="gan_images")
