# 1D toy regression problem 

**Author:** [Miquel Florensa](https://www.linkedin.com/in/miquel-florensa/)  
**Date:** 2023/03/15  
**Description:** This example shows how to use the a simple feedforward neural network to solve a 1D toy regression problem.  

<a href="https://github.com/lhnguyen102/cuTAGI/blob/main/python_examples/regression_runner.py" class="github-link">
  <div class="github-icon-container">
    <img src="../../images/GitHub-Mark.png" alt="GitHub" height="32" width="64">
  </div>
  <div class="github-text-container">
    Github Source code
  </div>
</a>

---

## 1. Setup

```python
from visualizer import PredictionViz

from python_examples.data_loader import RegressionDataLoader
from python_examples.model import RegressionMLP
from python_examples.regression import Regression
```

?>Notice that this modules are described [here](modules/modules.md) and the source code is in the *python_examples* directory, in case you have the modules in another directory you must change this paths.

## 2. Prepare the data

In this simple example we will use a 1D toy dataset. The data is generated from a function with a random noise. The goal is to learn the function from the data.

```python
# User-input
num_inputs = 1      # 1 explanatory variable
num_outputs = 1     # 1 predicted output
num_epochs = 50     # row for 50 epochs
x_train_file = "./data/toy_example/x_train_1D.csv"
y_train_file = "./data/toy_example/y_train_1D.csv"
x_test_file = "./data/toy_example/x_test_1D.csv"
y_test_file = "./data/toy_example/y_test_1D.csv"
```

**You can find the used data in the [toy_example data](https://github.com/lhnguyen102/cuTAGI/tree/main/data/toy_example) in the repository.*

?>We can plot the training data points and the trend line we want to learn.

![1D toy regression problem data](../../images/1D_toy_regression_data.png)


## 3. Create the model

We will use a FNN with a simple architecture as defined in the RegressionMLP class wich is suited for this basic regression problem. Find out more about the [RegressionMLP class](modules/models?id=regression-mlp-class).

```python
# Model
net_prop = RegressionMLP()
```

> If you want to use a different model, you can create your own class and make sure that it inherits from the NetProp class, more information in [models page](modules/models?id=netprop-class).

## 4. Load the data

We will make use of the [RegressionDataLoader](modules/data-loader?id=data-loader) class to load and process the data. The *process_data* function requires the input and output test and training files in a **csv** format.

```python
# Data loader
reg_data_loader = RegressionDataLoader(num_inputs=num_inputs,
                                       num_outputs=num_outputs,
                                       batch_size=net_prop.batch_size)
                                       
data_loader = reg_data_loader.process_data(x_train_file=x_train_file,
                                           y_train_file=y_train_file,
                                           x_test_file=x_test_file,
                                           y_test_file=y_test_file)
```

## 5. Create visualizer

In order to visualize the predictions of the regression we can use the PredictionViz class. This class will create a window with the true function, the predicted function and the confidence intervals.

```python
viz = PredictionViz(task_name="regression", data_name="toy1D")
```

> Learn more about  PredictionViz class [here](https://github.com/lhnguyen102/cuTAGI/blob/main/visualizer.py).

## 6. Train and evaluate the model

Using the [regression class](modules/regression?id=regression-class) that makes use of TAGI, we will train and test the model. When doing the prediction we can specify the standard deviation factor to calculate the confidence intervals.

```python
reg_task = Regression(num_epochs=num_epochs,
                      data_loader=data_loader,
                      net_prop=net_prop,
                      viz=viz)

reg_task.train()
reg_task.predict(std_factor=3)
```

## 7. Visualize the results

At the end of the execution the results will be printed in the console as seen below.

> MSE           :  1026.14  
> Log-likelihood: -5.89  

?> If you have created the visualizarion object and passed it to the regression object, a new window will pop up with the results.

![1D toy regression problem](../../images/1D_toy_regression.png)

**The black line is the true function, the red line is the predicted function and the red zone is the confidence intervals.*
