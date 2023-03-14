<!-------------------------------------------------------------------
File:         tutorial.md
Description:  FNN tutorial with 1D data
Authors:      Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet
Created:      March 02, 2023
Updated:      March 02, 2023
Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com & james.goulet@polymtl.ca
Copyright (c) 2023 Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
-------------------------------------------------------------------->

## Introduction

In this tutorial, we will see how to use pytagi to solve a simple regression problem. We will use a 1D toy dataset and a feedforward neural network (FNN) with a simple architecture.

## Define user input and data

In this simple example, we will use a 1D toy dataset. The dataset is composed of 10 training samples and 100 test samples and can be found in the [github repository](https://github.com/lhnguyen102/cuTAGI/tree/main/data/toy_example).

```python
# User-input
num_inputs = 1
num_outputs = 1
x_train_file = "./data/toy_example/x_train_1D.csv"
y_train_file = "./data/toy_example/y_train_1D.csv"
x_test_file = "./data/toy_example/x_test_1D.csv"
y_test_file = "./data/toy_example/y_test_1D.csv"
```

## Build Regression Model

We will use a FNN with a simple architecture. We will use the RegressionMLP class already defined and consistent with this basic regression problem (you can find the class implementation [here](modules/models?id=regression-mlp-class)).

```python
# Model
net_prop = RegressionMLP()
```

If you want to use a different model, you can define your own class and make sure that it inherits from the NetProp class, more information in [models page](modules/models?id=mlp-generic-class).

## Data loader

We will make use of the [RegressionDataLoader](modules/data-loader.md) class to load and process the data. The *process_data* function requires the input and output test and training files in a **csv** format.

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

## Train and test the model

Using the [regression class](modules/regression?id=regression-class) that makes use of TAGI, we will train and test the model. In order to perform the task we will also need to specify the number of epochs.

```python
# Optional: Visualize the test using visualizer.py
viz = PredictionViz(task_name="regression", data_name="toy1D")

num_epochs = 50

# Train and test
reg_task = Regression(num_epochs=num_epochs,
                      data_loader=data_loader,
                      net_prop=net_prop,
                      viz=viz)
reg_task.train()
reg_task.predict(std_factor=3)
```

**Learn more about  PredictionViz class [here](https://github.com/lhnguyen102/cuTAGI/blob/main/visualizer.py).*

## Results

Since we have created a vizualization object that we passed to the regression class, at the end of the execution we will be able to see the results of the prediction. The black line is the true function, the red line is the predicted function and the red zone is the confidence intervals.

![1D toy regression problem](../images/1D_toy_regression.png)
