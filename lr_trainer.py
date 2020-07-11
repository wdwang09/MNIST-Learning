from dataset import *
import numpy as np
import time

class LRTrainer:
    def __init__(self, train_img, train_label, test_img, test_label):
        self.trainImg = train_img
        self.trainLabel = train_label
        self.testImg = test_img
        self.testLabel = test_label

    def mini_batch(self, X, y, derive_func, batch_size=5000, learning_rate=0.001, epoch_nums=20):
        t1 = time.process_time()
        v = np.zeros((X.shape[1], y.shape[1]), dtype=np.float64)
        beta = np.zeros((X.shape[1], y.shape[1]), dtype=np.float64)
        for epoch in range(epoch_nums):
            print("epoch ", epoch+1, '/', epoch_nums, sep='')
            for batch_index in range(X.shape[0] // batch_size):
                X_batch = X[batch_index * batch_size:(batch_index+1) * batch_size]
                y_batch = y[batch_index * batch_size:(batch_index+1) * batch_size]
                v = 0.9 * v + learning_rate * derive_func(beta, X_batch, y_batch) / X_batch.shape[0]
                beta -= v  # momentum
                # beta -= learning_rate * derive_func(beta, X_batch, y_batch) / batch_size
            print("Accuracy:", self.accuracy(beta))
        t2 = time.process_time()
        print("Time: ", t2 - t1)
        return beta

    def accuracy(self, beta):
        test_num = self.testImg.shape[0]
        test_img_with_bias = np.hstack((np.ones((test_num, 1), dtype=np.uint8), self.testImg))
        res = np.dot(test_img_with_bias, beta)
        prediction = np.argmax(self.sigmoid(res), axis=1)
        ground_truth = np.argmax(self.testLabel, axis=1)

        correct = np.sum(prediction == ground_truth)
        return correct / test_num

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    def logistic_regression(self, batch_size=5000, learning_rate=0.001, epoch_nums=30):
        def lr_loss_function_der(beta, X, y):
            # print(y.shape)  # batch_size, 10
            res = np.dot(X, beta)
            # sig = 1 / (1 + np.exp(-1 * res))
            sig = self.sigmoid(res)
            return -np.dot(X.T, (y - sig))  # X.shape[1], 10

        train_img_with_bias = np.hstack((np.ones((self.trainImg.shape[0], 1), dtype=np.uint8), self.trainImg))
        beta = self.mini_batch(train_img_with_bias, self.trainLabel, lr_loss_function_der,
                               batch_size, learning_rate, epoch_nums)
        print("After training, the accuracy is", self.accuracy(beta))

    def logistic_regression_with_ridge_regression(self, batch_size=5000, learning_rate=0.001, epoch_nums=30):
        def lr_ridge_loss_function_der(beta, X, y):
            lambda_ = 0.5
            res = np.dot(X, beta)
            sig = self.sigmoid(res)
            return -2 * np.dot(X.T, (y - sig)) + 2 * lambda_ * beta  # X.shape[1], 10

        train_img_with_bias = np.hstack((np.ones((self.trainImg.shape[0], 1), dtype=np.uint8), self.trainImg))
        beta = self.mini_batch(train_img_with_bias, self.trainLabel, lr_ridge_loss_function_der,
                               batch_size, learning_rate, epoch_nums)
        print("After training, the accuracy is", self.accuracy(beta))

    def logistic_regression_with_lasso_regression(self, batch_size=5000, learning_rate=0.001, epoch_nums=30):
        def lr_lasso_loss_function_der(beta, X, y):
            lambda_ = 0.5
            lasso = np.zeros_like(beta)
            lasso[beta > 0] = 1
            lasso[beta < 0] = -1
            lasso[beta == 0] = 0
            res = np.dot(X, beta)
            sig = self.sigmoid(res)
            return -2 * np.dot(X.T, (y - sig)) + lambda_ * lasso  # X.shape[1], 10

        train_img_with_bias = np.hstack((np.ones((self.trainImg.shape[0], 1), dtype=np.uint8), self.trainImg))
        beta = self.mini_batch(train_img_with_bias, self.trainLabel, lr_lasso_loss_function_der,
                               batch_size, learning_rate, epoch_nums)
        print("After training, the accuracy is", self.accuracy(beta))


if __name__ == '__main__':
    # ds = ManualDataset(True, "MNIST", "MNIST")
    ds = TorchReaderDataset('./data')
    trainer = LRTrainer(ds.get_vector_train_img(is_normalized=True), ds.get_matrix_train_label(),
                        ds.get_vector_test_img(is_normalized=True), ds.get_matrix_test_label())
    trainer.logistic_regression(learning_rate=0.02, epoch_nums=20, batch_size=3000)
    print('----------------')
    trainer.logistic_regression_with_ridge_regression(learning_rate=0.02, epoch_nums=20, batch_size=3000)
    print('----------------')
    trainer.logistic_regression_with_lasso_regression(learning_rate=0.02, epoch_nums=20, batch_size=3000)

    # trainer = LRTrainer(ds.get_lda_vector_train_img(8), ds.get_matrix_train_label(),
    #                     ds.get_lda_vector_test_img(8), ds.get_matrix_test_label())
    # trainer.logistic_regression(learning_rate=0.05, epoch_nums=50)

    # trainer = LRTrainer(ds.get_pca_vector_train_img(2), ds.get_matrix_train_label(),
    #                     ds.get_pca_vector_test_img(2), ds.get_matrix_test_label())
    # trainer.logistic_regression(learning_rate=0.05, epoch_nums=50)
