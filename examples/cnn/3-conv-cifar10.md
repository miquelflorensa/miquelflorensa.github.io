# 3-Convolutional CIFAR10

**Author:** [Miquel Florensa](https://www.linkedin.com/in/miquel-florensa/)  
**Date:** 2023/05/05  
**Description:** This example shows how to train a convolutional neural network (CNN) to classify the CIFAR10 dataset.

<a href="https://github.com/lhnguyen102/cuTAGI/blob/main/python_examples/classification_runner.py" class="github-link">
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
from python_examples.model import ConvCifarMLP

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

In this example we will create a model of two convolutional layers, a batch size of 16 and will use hierarchical softmax for the classification task. Find out more about the [ConvCifarMLP class](modules/models?id=_3-conv-cifar10-classification-mlp-class) and all its parameters.

```python
# Model
net_prop = ConvCifarMLP()
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

| Model | Error Rate [%] |  | Hyperparameters |  |
| :---: | :---: | :---: | :---: | :---: |
|       | e = 1 | e = E | E | B |
| **TAGI** | 52.71 | 29.66 | 50 | 16 |
| BP | - | 23.5  | 150 | 128 |

?> The table above compares the classification accuracy with the results from [Wan et al.](http://proceedings.mlr.press/v28/wan13.pdf) where both approaches use the same CNN architecture with 3 convolutional layers (32-16-8) and a fully connected layer with 64 hidden units.
