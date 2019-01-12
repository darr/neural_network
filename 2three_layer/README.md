#Implementation of the feedforward neural network with cifar10 dataset and mnist dataset

cifar10 data web site.
```shell
http://www.cs.toronto.edu/~kriz/cifar.html
```

Download the binary vesion dataset.
```shell
CIFAR-10 binary version (suitable for C programs)   162 MB  c32a1d4ab5d03f1284b67883e8d87530
```

Download 'cifar-10-binary.tar.gz' to fold './data/'

```shell
cd ./data/
```
#first decompression the dataset 'cifar-10-binary.tar.gz' to the fold './cifar10_data/'

Run the shell code below:
```shell
tar -xzvf cifar-10-binary.tar.gz
```

Then will appear a fold 'cifar-10-batches-bin'.  
And the files in the fold:  
```shell
batches.meta.txt
data_batch_1.bin
data_batch_2.bin
data_batch_3.bin
data_batch_4.bin
data_batch_5.bin
readme.html
test_batch.bin
```

MNIST web site
```shell
http://yann.lecun.com/exdb/mnist/
```
There are four files.

```shell
train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
```
download the files.
#decompression the dataset to the fold './mnist_data/'
```shell
$ gunzip *.gz
```
output
```shell
t10k-images-idx3-ubyte
train-images-idx3-ubyte
t10k-labels-idx1-ubyte
train-labels-idx1-ubyte
```

Suppose you have the virtual environment,I use anaconda this time instead of virtualenv. Roughly seaking, I do not known how to install cudatoolkit which I need to be used accelerating computation by GPU.
It is easy for anaconda to install cudatoolkit.
```shell
conda install cudatoolkit
```
cudatoolkit installed by anaconda is different from Nvidia CUDA ToolKit despite that they have the same name.


#enviroment need by python2
```shell
pip install Pillow
pip install mysqlclient
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade numpy 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade matplotlib==2.2.2
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade numba
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade scipy
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade cudatoolkit  #install failed
```
#enviroment need by python3
```shell
pip install Pillow
#3.5 现在还不支持MySQLdb
pip install PyMySQL
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade numpy 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade matplotlib
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade numba
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade scipy
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade cudatoolkit  #install failed
```
