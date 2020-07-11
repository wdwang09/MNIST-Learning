import numpy as np
import os


class BaseDataset:
    def __init__(self):
        self.trainImg = None
        self.trainLabel = None
        self.testImg = None
        self.testLabel = None
        self.channels, self.rows, self.cols = 1, 28, 28
    
    def get_vector_train_img(self, is_normalized=True):
        if is_normalized:
            return (self.trainImg.astype(np.float64) - 128) / 128
        else:
            return self.trainImg

    def get_matrix_train_img_with_channel(self, is_normalized=True):
        tmp = self.get_vector_train_img(is_normalized)
        return tmp.reshape((tmp.shape[0], self.channels, self.rows, self.cols))

    def get_vector_test_img(self, is_normalized=True):
        if is_normalized:
            return (self.testImg.astype(np.float64) - 128) / 128
        else:
            return self.testImg

    def get_matrix_test_img_with_channel(self, is_normalized=True):
        tmp = self.get_vector_test_img(is_normalized)
        return tmp.reshape((tmp.shape[0], self.channels, self.rows, self.cols))

    def get_matrix_train_label(self):
        return self.trainLabel

    def get_vector_train_label(self):
        return np.argmax(self.trainLabel, axis=1)

    def get_matrix_test_label(self):
        return self.testLabel

    def get_vector_test_label(self):
        return np.argmax(self.testLabel, axis=1)

    def pca_projection(self, dim=100):
        origin = self.get_vector_train_img(True)
        sigma = np.dot(origin.transpose(), origin)  # / origin.shape[1] # You can divide it or not.
        U, _, _ = np.linalg.svd(sigma)
        return U[:, :dim]

    def lda_projection(self, dim=100):
        origin_imgs = self.get_vector_train_img(is_normalized=True)
        origin_labels = self.get_vector_train_label()
        labels = np.unique(origin_labels)
        images_dict = dict()
        mean_dict = dict()
        for label in labels:
            images_dict[label] = origin_imgs[origin_labels == label]
            mean_dict[label] = np.mean(images_dict[label], axis=0)
        mean_all = np.mean(origin_imgs, axis=0)

        origin_feature_size = origin_imgs.shape[1]
        Sw = np.zeros((origin_feature_size, origin_feature_size), dtype=np.float64)
        for label in labels:
            Sw += np.dot((images_dict[label]-mean_dict[label]).T,
                         images_dict[label]-mean_dict[label])

        Sb = np.zeros((origin_feature_size, origin_feature_size), dtype=np.float64)
        for label in labels:
            Sb += images_dict[label].shape[0] * np.dot((mean_dict[label]-mean_all).reshape(-1, 1),
                                                       (mean_dict[label]-mean_all).reshape(1, -1))

        # use pinv rather than inv to avoid singularity

        Sw_Sb = np.linalg.pinv(Sw).dot(Sb)
        U, _, _ = np.linalg.svd(Sw_Sb)
        return U[:, :dim]

    def get_pca_vector_train_img(self, dim=100, is_normalized=True):
        projection = self.pca_projection(dim)
        res = np.dot(self.get_vector_train_img(is_normalized=True), projection)
        if is_normalized:
            mean = np.mean(res, axis=0)
            std = np.std(res, axis=0)
            # mean = 0
            # std = 1
            # x_min, x_max = np.min(res, 0), np.max(res, 0)
            # res = (res - x_min) / (x_max - x_min)
            return (res - mean) / np.sqrt(std**2 + 0.001)
        return res

    def get_pca_vector_test_img(self, dim=100, is_normalized=True):
        projection = self.pca_projection(dim)
        res = np.dot(self.get_vector_test_img(is_normalized=True), projection)
        if is_normalized:
            mean = np.mean(res, axis=0)
            std = np.std(res, axis=0)
            # mean = 0
            # std = 1
            return (res - mean) / np.sqrt(std**2 + 0.001)
        return res

    def get_lda_vector_train_img(self, dim=100, is_normalized=True):
        projection = self.lda_projection(dim)
        res = np.dot(self.get_vector_train_img(is_normalized=True), projection)
        if is_normalized:
            mean = np.mean(res, axis=0)
            std = np.std(res, axis=0)
            return (res - mean) / np.sqrt(std**2 + 0.0001)
        return res

    def get_lda_vector_test_img(self, dim=100, is_normalized=True):
        projection = self.lda_projection(dim)  # same as train img
        res = np.dot(self.get_vector_test_img(is_normalized=True), projection)
        if is_normalized:
            mean = np.mean(res, axis=0)
            std = np.std(res, axis=0)
            return (res - mean) / np.sqrt(std**2 + 0.0001)
        return res

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
        # plot_embedding(X_tsne)

class ManualDataset(BaseDataset):
    def __init__(self, is_read_from_file=False, dataset_path="MNIST", output_numpy_dir=""):
        self.isReadFromFile = is_read_from_file
        self.path = dataset_path
        self.outputPath = output_numpy_dir
        self.trainImg = None
        self.trainLabel = None
        self.trainNum = 0
        self.testImg = None
        self.testLabel = None
        self.testNum = 0
        self.rows = 28
        self.cols = 28
        self.channels = 1
        self.__load()

    # def get_vector_train_img(self, is_normalized=True):
    #     if is_normalized:
    #         return (self.trainImg.astype(np.float64) - 128) / 128
    #     else:
    #         return self.trainImg

    # def get_matrix_train_img_with_channel(self, is_normalized=True):
    #     tmp = self.get_vector_train_img(is_normalized)
    #     return tmp.reshape((tmp.shape[0], self.channels, self.rows, self.cols))

    # def get_vector_test_img(self, is_normalized=True):
    #     if is_normalized:
    #         return (self.testImg.astype(np.float64) - 128) / 128
    #     else:
    #         return self.testImg

    # def get_matrix_test_img_with_channel(self, is_normalized=True):
    #     tmp = self.get_vector_test_img(is_normalized)
    #     return tmp.reshape((tmp.shape[0], self.channels, self.rows, self.cols))

    # def get_matrix_train_label(self):
    #     return self.trainLabel

    # def get_vector_train_label(self):
    #     return np.argmax(self.trainLabel, axis=1)

    # def get_matrix_test_label(self):
    #     return self.testLabel

    # def get_vector_test_label(self):
    #     return np.argmax(self.testLabel, axis=1)

    # def pca_projection(self, dim=100):
    #     origin = self.get_vector_train_img(True)
    #     sigma = np.dot(origin.transpose(), origin)  # / origin.shape[1] # You can divide it or not.
    #     U, _, _ = np.linalg.svd(sigma)
    #     return U[:, :dim]

    # def lda_projection(self, dim=100):
    #     origin_imgs = self.get_vector_train_img(is_normalized=True)
    #     origin_labels = self.get_vector_train_label()
    #     labels = np.unique(origin_labels)
    #     images_dict = dict()
    #     mean_dict = dict()
    #     for label in labels:
    #         images_dict[label] = origin_imgs[origin_labels == label]
    #         mean_dict[label] = np.mean(images_dict[label], axis=0)
    #     mean_all = np.mean(origin_imgs, axis=0)

    #     origin_feature_size = origin_imgs.shape[1]
    #     Sw = np.zeros((origin_feature_size, origin_feature_size), dtype=np.float64)
    #     for label in labels:
    #         Sw += np.dot((images_dict[label]-mean_dict[label]).T,
    #                      images_dict[label]-mean_dict[label])

    #     Sb = np.zeros((origin_feature_size, origin_feature_size), dtype=np.float64)
    #     for label in labels:
    #         Sb += images_dict[label].shape[0] * np.dot((mean_dict[label]-mean_all).reshape(-1, 1),
    #                                                    (mean_dict[label]-mean_all).reshape(1, -1))

    #     # use pinv rather than inv to avoid singularity

    #     Sw_Sb = np.linalg.pinv(Sw).dot(Sb)
    #     U, _, _ = np.linalg.svd(Sw_Sb)
    #     return U[:, :dim]

    # def get_pca_vector_train_img(self, dim=100, is_normalized=True):
    #     projection = self.pca_projection(dim)
    #     res = np.dot(self.get_vector_train_img(is_normalized=True), projection)
    #     if is_normalized:
    #         mean = np.mean(res, axis=0)
    #         std = np.std(res, axis=0)
    #         # mean = 0
    #         # std = 1
    #         # x_min, x_max = np.min(res, 0), np.max(res, 0)
    #         # res = (res - x_min) / (x_max - x_min)
    #         return (res - mean) / np.sqrt(std**2 + 0.001)
    #     return res

    # def get_pca_vector_test_img(self, dim=100, is_normalized=True):
    #     projection = self.pca_projection(dim)
    #     res = np.dot(self.get_vector_test_img(is_normalized=True), projection)
    #     if is_normalized:
    #         mean = np.mean(res, axis=0)
    #         std = np.std(res, axis=0)
    #         # mean = 0
    #         # std = 1
    #         return (res - mean) / np.sqrt(std**2 + 0.001)
    #     return res

    # def get_lda_vector_train_img(self, dim=100, is_normalized=True):
    #     projection = self.lda_projection(dim)
    #     res = np.dot(self.get_vector_train_img(is_normalized=True), projection)
    #     if is_normalized:
    #         mean = np.mean(res, axis=0)
    #         std = np.std(res, axis=0)
    #         return (res - mean) / np.sqrt(std**2 + 0.0001)
    #     return res

    # def get_lda_vector_test_img(self, dim=100, is_normalized=True):
    #     projection = self.lda_projection(dim)  # same as train img
    #     res = np.dot(self.get_vector_test_img(is_normalized=True), projection)
    #     if is_normalized:
    #         mean = np.mean(res, axis=0)
    #         std = np.std(res, axis=0)
    #         return (res - mean) / np.sqrt(std**2 + 0.0001)
    #     return res

    # def t_sne(self, visual_part, labels=None, title=None):
    #     from sklearn import manifold
    #     import matplotlib.pyplot as plt

    #     def plot_embedding(X, y=None, title=None):
    #         x_min, x_max = np.min(X, 0), np.max(X, 0)
    #         X = (X - x_min) / (x_max - x_min)

    #         # plt.figure()
    #         # ax = plt.subplot(111)
    #         if y is None:
    #             plt.scatter(X[:, 0], X[:, 1], marker='.')
    #         else:
    #             for i in range(X.shape[0]):
    #                 plt.text(X[i, 0], X[i, 1], str(y[i]),
    #                          color=plt.cm.Set1(y[i] / 10.),
    #                          fontdict={'weight': 'bold', 'size': 9})

    #         plt.xticks([]), plt.yticks([])
    #         if title is not None:
    #             plt.title(title)
    #         # plt.savefig(fname="graphs/{}.pdf".format(title.replace(' ', '_')), format="pdf")
    #         plt.show()

    #     print("Do T-SNE...")
    #     tsne = manifold.TSNE(n_components=2)  # , init='random', random_state=None)
    #     X_tsne = tsne.fit_transform(visual_part[:1000])
    #     print("Plotting...")
    #     Y = None
    #     if labels is not None:
    #         Y = labels[:1000]
    #     plot_embedding(X_tsne, Y, title)
    #     # plot_embedding(X_tsne)

    def __load(self):  # load_method: http://yann.lecun.com/exdb/mnist/

        # train img
        if self.isReadFromFile and os.path.exists(os.path.join(self.path, "train-images.idx3-ubyte.npy")):
            self.trainImg = np.load(os.path.join(self.path, "train-images.idx3-ubyte.npy"))
            self.trainNum = self.trainImg.shape[0]
        else:
            with open(os.path.join(self.path, "train-images.idx3-ubyte"), 'rb') as tif:
                tif.read(4)  # magic number 2051
                self.trainNum = int(tif.read(4).hex(), 16)  # numbers of train images
                rows = int(tif.read(4).hex(), 16)  # numbers of rows
                cols = int(tif.read(4).hex(), 16)  # numbers of cols
                # self.rows = rows
                # self.cols = cols
                self.trainImg = np.zeros((self.trainNum, rows * cols), dtype=np.uint8)
                for img_id in range(self.trainNum):
                    pic_str = tif.read(rows * cols).hex()
                    for i in range(rows * cols):
                            self.trainImg[img_id, i] = \
                                int(pic_str[i * 2: i * 2 + 2], 16)
                    # for i in range(rows):
                    #     for j in range(cols):
                    #         if self.trainImg[img_id, i * rows + j] > 0:
                    #             print('#', end='')
                    #         else:
                    #             print(' ', end='')
                    #     print()
                    # print()
                if self.outputPath != "":
                    if not os.path.exists(self.outputPath):
                        os.mkdir(self.outputPath)
                    np.save(os.path.join(self.outputPath, "train-images.idx3-ubyte"), self.trainImg)

        # test img
        if self.isReadFromFile and os.path.exists(os.path.join(self.path, "t10k-images.idx3-ubyte.npy")):
            self.testImg = np.load(os.path.join(self.path, "t10k-images.idx3-ubyte.npy"))
            self.testNum = self.testImg.shape[0]
        else:
            with open(os.path.join(self.path, "t10k-images.idx3-ubyte"), 'rb') as tif:
                tif.read(4)  # magic number 2051
                self.testNum = int(tif.read(4).hex(), 16)  # numbers of train images
                rows = int(tif.read(4).hex(), 16)  # numbers of rows
                cols = int(tif.read(4).hex(), 16)  # numbers of cols
                self.testImg = np.zeros((self.testNum, rows * cols), dtype=np.uint8)
                for img_id in range(self.testNum):
                    pic_str = tif.read(rows * cols).hex()
                    for i in range(rows * cols):
                            self.testImg[img_id, i] = \
                                int(pic_str[i * 2: i * 2 + 2], 16)
                    # for i in range(rows):
                    #     for j in range(cols):
                    #         if self.testImg[img_id, i * rows + j] > 0:
                    #             print('#', end='')
                    #         else:
                    #             print(' ', end='')
                    #     print()
                    # print()
                if self.outputPath != "":
                    if not os.path.exists(self.outputPath):
                        os.mkdir(self.outputPath)
                    np.save(os.path.join(self.outputPath, "t10k-images.idx3-ubyte"), self.testImg)

        # train label
        if self.isReadFromFile and os.path.exists(os.path.join(self.path, "train-labels.idx1-ubyte.npy")):
            self.trainLabel = np.load(os.path.join(self.path, "train-labels.idx1-ubyte.npy"))
        else:
            with open(os.path.join(self.path, "train-labels.idx1-ubyte"), 'rb') as tif:
                tif.read(4)  # magic number 2049
                tif.read(4)  # numbers of train images
                self.trainLabel = np.zeros((self.trainNum, 10), dtype=np.uint8)
                label_str = tif.read(self.trainNum).hex()
                for img_id in range(self.trainNum):
                    digit = int(label_str[img_id * 2:img_id * 2 + 2], 16)
                    self.trainLabel[img_id][digit] = 1
                if self.outputPath != "":
                    if not os.path.exists(self.outputPath):
                        os.mkdir(self.outputPath)
                    np.save(os.path.join(self.outputPath, "train-labels.idx1-ubyte"), self.trainLabel)

        # test label
        if self.isReadFromFile and os.path.exists(os.path.join(self.path, "t10k-labels.idx1-ubyte.npy")):
            self.testLabel = np.load(os.path.join(self.path, "t10k-labels.idx1-ubyte.npy"))
        else:
            with open(os.path.join(self.path, "t10k-labels.idx1-ubyte"), 'rb') as tif:
                tif.read(4)  # magic number 2049
                tif.read(4)  # numbers of test images
                self.testLabel = np.zeros((self.testNum, 10), dtype=np.uint8)
                label_str = tif.read(self.testNum).hex()
                for img_id in range(self.testNum):
                    digit = int(label_str[img_id * 2:img_id * 2 + 2], 16)
                    self.testLabel[img_id][digit] = 1
                if self.outputPath != "":
                    if not os.path.exists(self.outputPath):
                        os.mkdir(self.outputPath)
                    np.save(os.path.join(self.outputPath, "t10k-labels.idx1-ubyte"), self.testLabel)


class TorchReaderDataset(BaseDataset):
    def __init__(self, dataset_path="./data"):
        self.path = dataset_path
        self.trainImg = None
        self.trainLabel = None
        self.testImg = None
        self.testLabel = None
        self.rows = 28
        self.cols = 28
        self.channels = 1
        self.__load()
    
    def __load(self):
        import torch
        import torchvision
        with torch.no_grad():
            self.trainset = torchvision.datasets.MNIST(root=self.path, train=True,
                                                    download=True)
            self.testset = torchvision.datasets.MNIST(root=self.path, train=False,
                                                    download=True)
            # self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=100000000000,
            #                                             shuffle=False, num_workers=2)
            # self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100000000000,
            #                                             shuffle=False, num_workers=2)
            
            self.trainImg = self.trainset.data.view(-1, self.channels * self.rows * self.cols).cpu().numpy()
            trainLabel_ = self.trainset.targets.cpu().numpy()
            self.testImg = self.testset.data.view(-1, self.channels * self.rows * self.cols).cpu().numpy()
            testLabel_ = self.testset.targets.cpu().numpy()

            self.trainLabel = np.zeros((self.trainImg.shape[0], 10), dtype=np.uint8)
            self.testLabel = np.zeros((self.testImg.shape[0], 10), dtype=np.uint8)
            self.trainLabel[range(len(trainLabel_)), trainLabel_] = 1
            self.testLabel[range(len(testLabel_)), testLabel_] = 1

            # self.transform = torchvision.transforms.(
            # [transform])
            # for (data, target) in self.trainloader:
            #     self.trainImg = data.view(-1, self.channels * self.rows * self.cols).detach().cpu().numpy()
            #     self.trainLabel = target.detach().cpu().numpy()
            
            # for (data, target) in self.testloader:
            #     self.testImg = data.view(-1, self.channels * self.rows * self.cols).detach().cpu().numpy()
            #     self.testLabel = target.detach().cpu().numpy()

            # self.trainLabel = self.trainset.train_labels.clone().detach()
            # self.testLabel = self.testset.test_labels.clone().detach()

            # self.trainImg = ((self.trainset.train_data.to(torch.float32) - 128) / 128).view(-1, 1, 28, 28)
            # self.testImg = ((self.testset.test_data.to(torch.float32) - 128) / 128).view(-1, 1, 28, 28)



if __name__ == '__main__':
    # 先把MNIST全部解压，并放在第二个参数"MNIST"文件夹下
    # 第一次读会读的很慢，所以如果第三个参数不为""时会把产生的矩阵存下来，存在第三个参数"MNIST"下
    # 第一个参数：是否从文件中读取矩阵（如果有矩阵被保存在第二个参数），而不是重新产生矩阵（测试用，现在可忽略这个参数，设为True即可）
    # ds = Dataset(True, "MNIST", "MNIST")

    ds = TorchReaderDataset("./data")
    # img_id = 9999
    # for i in range(28):
    #     for j in range(28):
    #         if ds.testImg[img_id, i * 28 + j] > 0:
    #             print('#', end='')
    #         else:
    #             print(' ', end='')
    #     print()
    # print(ds.testLabel[img_id])
    # print()
    # print(ds.get_vector_train_img(True)[0])

    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=5)
    # ds.t_sne(pca.fit_transform(ds.get_vector_train_img()), ds.get_vector_train_label())
    # ds.t_sne(ds.get_pca_vector_train_img(15), ds.get_vector_train_label(), "PCA with 15 Dimensions")
    # ds.t_sne(ds.get_pca_vector_train_img(30), ds.get_vector_train_label(), "PCA with 30 Dimensions")
    # ds.t_sne(ds.get_vector_train_img(), ds.get_vector_train_label(), "without PCA")

    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    # lda = LDA(n_components=9)
    # ds.t_sne(lda.fit_transform(ds.get_vector_train_img(), ds.get_vector_train_label()), ds.get_vector_train_label())

    ds.t_sne(ds.get_lda_vector_train_img(9), ds.get_vector_train_label(), "LDA with 9 Dimensions")
    ds.t_sne(ds.get_pca_vector_train_img(9), ds.get_vector_train_label(), "PCA with 9 Dimensions")
