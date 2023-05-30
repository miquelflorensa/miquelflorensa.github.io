# Classifier

The `Classifier` class is responsible for performing image classification using the TAGI algorithm.

## Constructor

### `__init__(self, num_epochs: int, data_loader: dict, net_prop: NetProp, num_classes: int)`

- Parameters:
  - `num_epochs` (int): The number of training epochs.
  - `data_loader` (dict): A dictionary containing the data loader for training and testing data.
  - `net_prop` (NetProp): An instance of the `NetProp` class specifying the network properties.
  - `num_classes` (int): The number of classes for the classification task.

- Returns: None

## Properties

### `num_classes`

- Description: Get the number of classes.

## Methods

### `train(self)`

- Description: Train the network using TAGI.

### `predict(self)`

- Description: Make predictions using TAGI.

### `train_one_hot(self)`

- Description: Train the network using TAGI with one-hot encoded labels.

### `predict_one_hot(self)`

- Description: Make predictions using TAGI with one-hot encoded labels.

### `init_inputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]`

- Description: Initialize the covariance matrix for inputs.

- Parameters:
  - `batch_size` (int): The batch size.

- Returns:
  - `Tuple[np.ndarray, np.ndarray]`: A tuple containing the input covariance matrices.

### `init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]`

- Description: Initialize the covariance matrix for outputs.

- Parameters:
  - `batch_size` (int): The batch size.

- Returns:
  - `Tuple[np.ndarray, np.ndarray]`: A tuple containing the output covariance matrices.

