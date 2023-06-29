# Regression class

The `Regression` class is responsible for performing regression using the TAGI algorithm.

<a href="https://github.com/lhnguyen102/cuTAGI/blob/main/python_examples/regression.py" class="github-link">
  <div class="github-icon-container">
    <img src="../images/GitHub-Mark.png" alt="GitHub" height="32" width="64">
  </div>
  <div class="github-text-container">
    Github Source code
  </div>
</a>


## Attributes

- `utils`: An instance of the `Utils` class.
- `num_epochs`: The number of epochs for training.
- `data_loader`: A dictionary containing the data loader.
- `net_prop`: An instance of the `NetProp` class representing the network properties.
- `network`: An instance of the `TagiNetwork` class.
- `dtype`: The data type (default: `np.float32`).
- `viz`: An optional instance of the `PredictionViz` class for visualization.

## *constructor* method

> Constructor for the Regression class.

```python
def __init__(self, num_epochs: int, data_loader: dict, net_prop: NetProp, dtype=np.float32, viz: Union[PredictionViz, None] = None) -> None:
```

**Parameters**
- `num_epochs`: An integer representing the number of epochs for training.
- `data_loader`: A dictionary containing the data loader.
- `net_prop`: An instance of the `NetProp` class representing the network properties.
- `dtype`: The data type (default: `np.float32`).
- `viz`: An optional instance of the `PredictionViz` class for visualization.

## *train* method

```python
def train(self) -> None:
    """Train the network using TAGI"""
```

Trains the network using TAGI algorithm. It performs the following steps:
1. Initializes inputs and outputs.
2. Performs training iterations for each epoch.
3. Updates the network parameters and hidden states.
4. Computes the loss (mean squared error) and displays the progress.

> See [TAGI](https://www.jmlr.org/papers/volume22/20-1009/20-1009.pdf) paper for more information.

## *predict* method

```python
def predict(self, std_factor: int = 1) -> None:
    """Make prediction using TAGI"""
```

Makes predictions using TAGI algorithm. It performs the following steps:
1. Initializes inputs.
2. Makes predictions using the trained network.
3. Unnormalizes the predictions.
4. Computes the mean squared error and log-likelihood of the predictions.
5. Prints the results.
 
> See [TAGI](https://www.jmlr.org/papers/volume22/20-1009/20-1009.pdf) paper for more information.

## *compute_derivatives* method

```python
def compute_derivatives(self, layer: int = 0, truth_derv_file: Union[None, str] = None) -> None:
    """Compute derivative of a given layer"""
```

Computes the derivative of a given layer in the network. It performs the following steps:
1. Initializes inputs.
2. Computes the derivatives using the trained network.
3. Unnormalizes the inputs.
4. Optionally plots the predictions against the ground truth derivatives.

## *init_inputs* method

```python
def init_inputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize the covariance matrix for inputs"""
```

Initializes the covariance matrix for inputs. It returns the initialized covariance matrices.

**Parameters**
- `batch_size`: An integer representing the batch size.

**Returns**
- A tuple containing the initialized covariance matrices for inputs.

## *init_outputs* method

```python
def init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize the covariance matrix for outputs"""
```

Initializes the covariance matrix for outputs. It returns the initialized covariance matrices.

**Parameters**
- `batch_size`: An integer representing the batch size.

**Returns**
- A tuple containing the initialized covariance matrices for outputs.
