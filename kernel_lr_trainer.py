from dataset import *
import numpy as np


class KernelBasedLRTrainer:
    def __init__(self, train_img, train_label, test_img, test_label):
        self.trainImg = train_img
        self.testImg = test_img
        self.trainLabel = train_label
        self.testLabel = test_label

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def gauss_kernel(x1, x2):
        K = np.zeros((x1.shape[0], x2.shape[0]), dtype=np.float64)
        gamma = 0.05  # 依靠调参使预测结果尽可能大（需要根据测试集的分布范围决定）
        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                # K[i, j] = np.dot(x1[i], x2[j])**3
                K[i, j] = np.exp(-gamma * (np.linalg.norm(x1[i] - x2[j], ord=2) ** 2))
        return K

    @staticmethod
    def poly_kernel(x1, x2):
        K = np.zeros((x1.shape[0], x2.shape[0]), dtype=np.float64)
        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                K[i, j] = np.dot(x1[i], x2[j])**4
        return K

    def choose_kernel(self, kernel_name):
        if kernel_name == "rbf":
            return self.gauss_kernel
        elif kernel_name == "poly":
            return self.poly_kernel
        else:
            raise NameError("The parameter `kernel_name` should be: {}.".format(["rbf", "poly"]))

    def calculate_kernel_and_label_for_training_and_testing(self, kernel_name, train_number=2000, test_number=2000):
        kernel = self.choose_kernel(kernel_name)
        train_img = self.trainImg[:train_number]
        train_label = self.trainLabel[:train_number]
        test_img = self.testImg[:test_number]
        test_label = self.testLabel[:test_number]
        print("Calculate {} kernel for training and testing...".format(kernel_name))
        K_train = kernel(train_img, train_img)
        print("K_train is calculated.")
        K_test = kernel(test_img, train_img)
        print("K_test is calculated.")
        return K_train, train_label, K_test, test_label

    def kernel_based_logistic_regression(self, kernel_name="rbf", train_number=2000, test_number=2000):
        def kernel_based_lr_loss_function_der(c, K, y):
            res = np.dot(K, c)
            sig = self.sigmoid(res)
            return -2 * np.dot(K.transpose(), (y - sig))

        K, sub_train_y, K_test, sub_test_y = self.calculate_kernel_and_label_for_training_and_testing(
            kernel_name, train_number, test_number)
        c = self.kernel_gradient_descent(K, sub_train_y, K_test, sub_test_y,
                                         kernel_based_lr_loss_function_der, 0.01, 100)
        print("After training, the accuracy is", self.kernel_accuracy(K_test, sub_test_y, c))

    def kernel_based_logistic_regression_with_ridge_regression(self, kernel_name="rbf",
                                                               train_number=2000, test_number=2000):
        def kernel_based_lr_ridge_loss_function_der(c, K, y):
            lambda_ = 0.5
            res = np.dot(K, c)
            sig = self.sigmoid(res)
            return -2 * np.dot(K.transpose(), (y - sig)) + 2 * lambda_ * c
        K, sub_train_y, K_test, sub_test_y = self.calculate_kernel_and_label_for_training_and_testing(
            kernel_name, train_number, test_number)
        c = self.kernel_gradient_descent(K, sub_train_y, K_test, sub_test_y,
                                         kernel_based_lr_ridge_loss_function_der, 0.01, 100)
        print("After training, the accuracy is", self.kernel_accuracy(K_test, sub_test_y, c))

    def kernel_based_logistic_regression_with_lasso_regression(self, kernel_name="rbf",
                                                               train_number=2000, test_number=2000):
        def kernel_based_lr_lasso_loss_function_der(c, K, y):
            lambda_ = 0.5
            lasso = np.zeros_like(c)
            lasso[c > 0] = 1
            lasso[c < 0] = -1
            lasso[c == 0] = 0
            res = np.dot(K, c)
            sig = self.sigmoid(res)
            return -2 * np.dot(K.transpose(), (y - sig)) + lambda_ * lasso
        K, sub_train_y, K_test, sub_test_y = self.calculate_kernel_and_label_for_training_and_testing(
            kernel_name, train_number, test_number)
        c = self.kernel_gradient_descent(K, sub_train_y, K_test, sub_test_y,
                                         kernel_based_lr_lasso_loss_function_der, 0.01, 100)
        print("After training, the accuracy is", self.kernel_accuracy(K_test, sub_test_y, c))

    def kernel_gradient_descent(self, K, y, K_test, y_test, derive_func, learning_rate=0.001, epoch_nums=100):
        beta = np.zeros((K.shape[1], 10), dtype=np.float64)
        for epoch in range(epoch_nums):
            beta -= learning_rate * derive_func(beta, K, y)
            if epoch % 5 == 0 or epoch + 1 == epoch_nums:
                print("epoch ", epoch + 1, '/', epoch_nums, sep='')
                print("Accuracy:", self.kernel_accuracy(K_test, y_test, beta))
        return beta

    def kernel_accuracy(self, K_test, y_test, beta):
        test_num = K_test.shape[0]
        res = np.dot(K_test, beta)
        prediction = np.argmax(self.sigmoid(res), axis=1)
        ground_truth = np.argmax(y_test, axis=1)

        correct = np.sum(prediction == ground_truth)
        return correct / test_num


if __name__ == '__main__':
    # ds = ManualDataset(True, "MNIST", "MNIST")
    ds = TorchReaderDataset('./data')
    trainer = KernelBasedLRTrainer(ds.get_vector_train_img(is_normalized=True), ds.get_matrix_train_label(),
                                   ds.get_vector_test_img(is_normalized=True), ds.get_matrix_test_label())
    trainer.kernel_based_logistic_regression("rbf", train_number=500, test_number=500)
    # trainer.kernel_based_logistic_regression_with_ridge_regression("rbf", train_number=1000, test_number=1000)
    # trainer.kernel_based_logistic_regression_with_lasso_regression("rbf", train_number=1000, test_number=1000)
