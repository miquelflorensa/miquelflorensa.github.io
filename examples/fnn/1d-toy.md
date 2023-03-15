# 1D toy regression problem 

**Author:** [Miquel Florensa](https://www.linkedin.com/in/miquel-florensa-630669182/)  
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

```python
# User-input
num_inputs = 1
num_outputs = 1
num_epochs = 50
x_train_file = "./data/toy_example/x_train_1D.csv"
y_train_file = "./data/toy_example/y_train_1D.csv"
x_test_file = "./data/toy_example/x_test_1D.csv"
y_test_file = "./data/toy_example/y_test_1D.csv"
```

**You can find the used data in the [toy_example data](https://github.com/lhnguyen102/cuTAGI/tree/main/data/toy_example) in the repository.*

## 3. Create the model

```python
# Model
net_prop = RegressionMLP()
```

> Find out more about the [RegressionMLP class](modules/models?id=regression-mlp-class).

## 4. Load the data

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

> More information about the [DataLoader class](modules/data-loader?id=data-loader). 

## 5. Create visualizer

```python
viz = PredictionViz(task_name="regression", data_name="toy1D")
```

> Learn more about  PredictionViz class [here](https://github.com/lhnguyen102/cuTAGI/blob/main/visualizer.py).

## 6. Create the regression object

```python
reg_task = Regression(num_epochs=num_epochs,
                      data_loader=data_loader,
                      net_prop=net_prop,
                      viz=viz)
```

> Find out more about the [Regression class](modules/regression?id=regression-class).

## 6. Train and evaluate the model

```python
reg_task.train()

reg_task.predict(std_factor=3)
```

## 7. Visualize the results

> MSE           :  1026.14  
> Log-likelihood: -5.89  

?> If you have created the visualizarion object and passed it to the regression object, a new window will pop up with the results.

![1D toy regression problem](../../images/1D_toy_regression.png)

**The black line is the true function, the red line is the predicted function and the red zone is the confidence intervals.*
