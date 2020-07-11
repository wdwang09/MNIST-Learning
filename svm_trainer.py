from dataset import *
import numpy as np
from sklearn.svm import SVC
import time


class SVMTrainer:
    def __init__(self, train_img, train_label, test_img, test_label):
        self.trainImg = train_img
        self.trainLabel = train_label
        self.testImg = test_img
        self.testLabel = test_label

    def support_vector_machine(self, kernel='rbf', train_number=6000):
        # Param train_number SHOULDN'T be too large, which will affect the training speed.

        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        # train_label = np.argmax(self.trainLabel, axis=1)
        # test_label = np.argmax(self.testLabel, axis=1)
        clf = SVC(C=1.0, kernel=kernel)
        print("SVM kernel is \"{}\".".format(kernel))
        t1 = time.process_time()
        clf.fit(self.trainImg[:train_number], self.trainLabel[:train_number])
        print("Training Time: {}".format(time.process_time() - t1))
        print("Accuracy:", clf.score(self.testImg, self.testLabel))
        print("Training + Testing Time: {}".format(time.process_time() - t1))


if __name__ == '__main__':
    # ds = ManualDataset(True, "MNIST", "MNIST")
    ds = TorchReaderDataset('./data')

    trainer = SVMTrainer(ds.get_vector_train_img(is_normalized=True), ds.get_vector_train_label(),
                         ds.get_vector_test_img(is_normalized=True), ds.get_vector_test_label())
    trainer.support_vector_machine('linear')
    # trainer.support_vector_machine('poly')
    # trainer.support_vector_machine('sigmoid')
    # trainer.support_vector_machine('rbf')
