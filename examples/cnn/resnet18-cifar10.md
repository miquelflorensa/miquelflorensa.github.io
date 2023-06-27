# ResNet18 CIFAR10

**Author:** [Miquel Florensa](https://www.linkedin.com/in/miquel-florensa/)  
**Date:** 2023/05/05  
**Description:** This example shows how to train a residual network to classify the CIFAR10 dataset.

<a href="https://github.com/miquelflorensa/miquelflorensa.github.io/blob/main/code/resnet_cifar10_classification_runner.py" class="github-link">
  <div class="github-icon-container">
    <img src="../../images/GitHub-Mark.png" alt="GitHub" height="32" width="64">
  </div>
  <div class="github-text-container">
    Github Source code
  </div>
</a>

---

## 1. Setup

We first import the required modules: the classifier, the data loader and the model.

```python
from python_examples.classification import Classifier
from python_examples.data_loader import ClassificationDataloader
from pytagi import NetProp

```

?>Notice that this modules are described [here](modules/modules.md) and the source code is in the *python_examples* directory, in case you have the modules in another directory you must change this paths.

## 2. Prepare the data

We define the number of epochs and the paths to the data. Notice that the data is in ubyte format and divided in four files: the training images, the training labels, the test images and the test labels.

```python
num_epochs = 50
x_train_file = "./data/cifar/x_train.csv"
y_train_file = "./data/cifar/y_train.csv"
x_test_file = "./data/cifar/x_test.csv"
y_test_file = "./data/cifar/y_test.csv"
```

**You can find the used data in the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) website.*

## 3. Create the model

In this example we are going to use a [Resnet18 architecture](https://arxiv.org/abs/1512.03385).

```python
class ResnetCifarMLP(NetProp):
    """Multi-layer perceptron for cifar classificaiton."""

    def __init__(self) -> None:
        super().__init__()
        #-------------------#Input-----------------#Stage1-------------------------#Stage2-------------------------#Stage3-------------------------#Stage4-------------------------#Output---
        self.layers =       [2,     2,      4,      2,      4,      2,      4,      2,      4,      2,      4,      2,      4,      2,      4,      2,      4,      2,      4,      1,     1]
        self.nodes =        [3072,  0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,    256,    11]
        self.kernels =      [7,     1,      3,      1,      3,      1,      3,      1,      3,      1,      3,      1,      3,      1,      3,      1,      3,      1,      3,      1,     1]
        self.strides =      [1,     0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,     0]
        self.widths =       [256, 256,    128,    128,     64,     64,     32,     32,     32,     32,     16,     16,      8,      8,      8,      4,      4,      4,      4,      1,     1]
        self.heights =      [256, 256,    128,    128,     64,     64,     32,     32,     32,     32,     16,     16,      8,      8,      8,      4,      4,      4,      4,      1,     1]
        self.filters =      [3,    64,     64,     64,     64,     64,     64,    128,    128,    128,    256,    256,    256,    256,    256,    512,    512,    512,     512,   512,     1]
        self.pads =         [1,     0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,     0]
        self.pad_types =    [1,     0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,     0]
        self.activations =  [0,     4,      0,      4,      0,      4,      0,      4,      0,      4,      0,      4,      0,      4,      0,      4,      0,      4,      0,      4,    12]
        self.shortcuts =    [-1,    3,     -1,      6,     -1,      8,     -1,     10,     -1,     12,     -1,     14,     -1,     16,     -1,     18,     -1,     20,     -1,     -1,    -1]
        self.batch_size = 16
        self.sigma_v = 1
        self.sigma_v_min = 0.2
        self.decay_factor_sigma_v = 0.975
        self.is_idx_ud = True
        self.multithreading = True
        self.init_method: str = "He"
        self.device = "cuda"
```

```python
# Model
net_prop = ResnetCifarMLP()
```

## 4. Load the data

The next step is to load the data. We will use the [ClassificationDataloader class](modules/data-loader?id=data-loader) to load the data and we will pass the batch size and the data paths to the class.

```python
# Data loader
class_data_loader = ClassificationDataloader(batch_size=net_prop.batch_size)
data_loader = class_data_loader.process_data(x_train_file=x_train_file,
                                            y_train_file=y_train_file,
                                            x_test_file=x_test_file,
                                            y_test_file=y_test_file)
```

## 5. Create the classification object

Once we processed the data, we can create the classifier object. We will pass the number of epochs, the data loader, the network properties and the number of classes to the class. In this case the number of classes is 10 because we are classifying the CIFAR10 dataset.

```python
# Train and test
clas_task = Classifier(num_epochs=num_epochs,
                      data_loader=data_loader,
                      net_prop=net_prop,
                      num_classes=10)
```

> Find out more about the [Classifier class](modules/classifier.md).

## 6. Train and evaluate the model

Finally, we can train and evaluate the model. We will call the train and predict methods of the classifier object.

```python
  clas_task.train()
  clas_task.predict()
```

## 7. Results

In this section we will see the performace of the model using cuTAGI and we will compare the results with the results of a backpropagation model.

|  Model   | Error Rate [%] |       | Hyperparameters |       |
| :------: | :------------: | :---: | :-------------: | :---: |
|          |     e = 1      | e = E |        E        |   B   |
| **TAGI** |      72.4      | 21.5  |       50        |  16   |
|    BP    |       -        | 14.0  |       160       |  128  |

?> The table above compares the classification accuracy with the results from [Osawa et al.](https://www.researchgate.net/publication/333650027_Practical_Deep_Learning_with_Bayesian_Principles) where they use a Resnet18 trained with backpropagation.
