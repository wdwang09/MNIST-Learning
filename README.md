# MNIST Learning with Different Methods



## 需要使用的库

代码只使用常见的基本库，大部分核心方法均由自己实现

* NumPy （矩阵运算）
* Matplotlib （作图）
* scikit-learn (sklearn) （SVM与特征分析等）
* PyTorch & torchvision （神经网络相关程序需要）



## 读入数据集

使用MNIST数据集，读取方法存放至`dataset.py`下，其它文件均需导入`dataset.py`下规定的数据集方可运行。

`dataset.py`提供了两种不同的数据集读取方式，一种是手动读入原始数据集（`ManualDataset`），一种是借助PyTorch进行读入（`TorchReaderDataset`）

对于`TorchReaderDataset`（推荐）：

1. 数据集的使用方法：`ds = TorchReaderDataset('./data')`，参数为Torch下载数据集的路径，下载后参数所在文件夹下会出现`MNIST`子文件夹。
2. 如果参数指定的路径已经存在数据集，则不会再次下载数据集。

对于`ManualDataset`（不推荐）：

1. 数据集的使用方法：`ds = ManualDataset(True, "MNIST", "MNIST")`，参数的说明见下。
2. 先把MNIST全部解压（不可以是gz文件），并放在第二个参数"MNIST"文件夹下。
3. 第一次读会读的很慢，所以如果第三个参数不为空字符串（""）时会把产生的矩阵存下来，存在第三个参数"MNIST"下。
4. 第一个参数：是否从文件中读取矩阵（如果有矩阵被保存在第二个参数），而不是从原始数据重新产生矩阵（设为True即可）。



对于训练数据，提供了两种不同的格式进行读入：n为样本个数，一种是`n * 784` (`get_vector_t..._img`)，另一种是`n * 1 * 28 * 28` (`get_matrix_t..._img_with_channel`)。

对于测试数据，提供了两种不同的格式进行读入：n为样本个数，一种是`n * 1` (`get_vector_t..._img`)，另一种是`n * 10` (其中一项是1，其它项是0) (`get_matrix_t..._label`)。



## 使用不同方法实现机器学习

实现了各种方法，根据需求按照main函数中已提供（或已注释）的写法进行修改即可：

非深度学习方法（不需要GPU（同时无GPU支持），Logistic Regression基于NumPy）：

* Logistic Regression: `lr_trainer.py` （包含Lasso和Ridge Regression）
* Kernel-Based Logistic Regression: `kernel_lr_trainer.py` （包含Lasso和Ridge Regression）
* SVM: `svm_trainer.py`

深度学习方法（自动检测CUDA环境进行CPU或GPU运算，使用PyTorch）：
* NN: `nn_trainer.py`
* GAN: `gan_trainer.py`
* VAE: `vae_trainer.py`



## 数据降维与特征分析

PCA和LDA的实现均位于`dataset.py`提供的数据集类中，使用API对训练或测试数据使用PCA和LDA：
```python
ds.get_pca_vector_train_img(9)  # 9为降维后的维度
ds.get_pca_vector_test_img(9)
ds.get_lda_vector_train_img(9)
ds.get_lda_vector_test_img(9)
```

t-SNE方法内嵌于需要进行特征分析的py文件中，实现方法可参考`dataset.py`中的`t_sne`方法。

神经网络中间层的t-SNE特征分析详见`nn_visualization_trainer.py`，大部分代码与`nn_trainer.py`相同。


