# TimeSeriesForecaster class

The `TimeSeriesForecaster` class is responsible for time series forecasting using the TAGI algorithm.

## Attributes

- `num_epochs`: An integer representing the number of epochs for training.
- `data_loader`: A dictionary containing the data loader.
- `net_prop`: An instance of the `NetProp` class representing the network properties.
- `network`: An instance of the `TagiNetwork` class.
- `viz`: An optional instance of the `PredictionViz` class for visualization.
- `dtype`: The data type (default: `np.float32`).

## *constructor* method

```python
def __init__(self, num_epochs: int, data_loader: dict, 
             net_prop: NetProp, param: Union[Param, None] = None, 
             viz: Union[PredictionViz, None] = None, dtype=np.float32) -> None:
```

**Parameters**
- `num_epochs`: An integer representing the number of epochs for training.
- `data_loader`: A dictionary containing the data loader.
- `net_prop`: An instance of the `NetProp` class representing the network properties.
- `param`: An optional instance of the `Param` class for setting network parameters (default: `None`).
- `viz`: An optional instance of the `PredictionViz` class for visualization (default: `None`).
- `dtype`: The data type (default: `np.float32`).

## *train* method

```python
def train(self) -> None:
    """Train LSTM network"""
```

Trains the LSTM network using the TAGI algorithm. It performs the following steps:
1. Initializes inputs and outputs.
2. Performs training iterations for each epoch.
3. Updates the hidden states, network parameters, and loss.
4. Displays the training progress.

## *predict* method

```python
def predict(self) -> None:
    """Make prediction for time series using TAGI"""
```

Makes predictions for time series using the TAGI algorithm. It performs the following steps:
1. Initializes inputs.
2. Makes predictions using the trained network.
3. Unnormalizes the predictions.
4. Computes the mean squared error, log-likelihood, and prints the results.
5. If visualization is enabled, plots the predictions.

## *init_inputs* method

```python
def init_inputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize the covariance matrix for inputs"""
```

**Parameters**
- `batch_size`: An integer representing the batch size.

**Returns**
- A tuple containing two numpy arrays: `Sx_batch` and `Sx_f_batch`.

## *init_outputs* method

```python
def init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize the covariance matrix for outputs"""
```

**Parameters**
- `batch_size`: An integer representing the batch size.

**Returns**
- A tuple containing two numpy arrays: `V_batch` and `ud_idx_batch`.