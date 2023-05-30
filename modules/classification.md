# The Classifier class

The `Classifier` class is responsible for performing image classification using the TAGI algorithm.

<a href="https://github.com/lhnguyen102/cuTAGI/blob/main/python_examples/classification.py" class="github-link">
  <div class="github-icon-container">
    <img src="../images/GitHub-Mark.png" alt="GitHub" height="32" width="64">
  </div>
  <div class="github-text-container">
    Github Source code
  </div>
</a>


## Attributes

- `hr_softmax`: Instance of the HierarchicalSoftmax class.
- `utils`: Instance of the Utils class.
- `num_epochs`: Number of epochs for training.
- `data_loader`: Dictionary containing data loaders.
- `net_prop`: Instance of the NetProp class representing the network properties.
- `num_classes`: Number of classes.
- `network`: Instance of the TagiNetwork class representing the network.

## *constructor* method

> Constructor for the Classifier class.

```python
def __init__(self, num_epochs: int, data_loader: dict, net_prop: NetProp,
             num_classes: int) -> None:
```

**Parameters**
- `num_epochs`: An integer representing the number of epochs for training.
- `data_loader`: A dictionary containing data loaders.
- `net_prop`: An instance of the NetProp class representing the network properties.
- `num_classes`: An integer representing the number of classes.

## *num_classes* getter method

```python
@property
def num_classes(self) -> int:
    """Get number of classes"""
```

**Returns**
- `int`: The number of classes.

## *num_classes* setter method

```python
@num_classes.setter
def num_classes(self, value: int) -> None:
    """Set number of classes"""
```

**Parameters**
- `value`: An integer representing the number of classes.

## *train* method

```python
def train(self) -> None:
    """Train the network using TAGI"""
```

> See [TAGI](https://www.jmlr.org/papers/volume22/20-1009/20-1009.pdf) paper for more information.

## *predict* method

```python
def predict(self) -> None:
    """Make prediction using TAGI"""
```

> See [TAGI](https://www.jmlr.org/papers/volume22/20-1009/20-1009.pdf) paper for more information.

## *train_one_hot* method

```python
def train_one_hot(self) -> None:
    """Train the network using TAGI with one-hot encoded labels"""
```

## *predict_one_hot* method

```python
def predict_one_hot(self) -> None:
    """Make predictions using TAGI with one-hot encoded labels."""
```

## *init_inputs* method

```python
def init_inputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Initnitalize the covariance matrix for inputs"""
```

**Parameters**
  - `batch_size` (int): The batch size.

**Returns**
  - `Tuple[np.ndarray, np.ndarray]`: A tuple containing the input covariance matrices.

## *init_outputs* method

```python
def init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Initnitalize the covariance matrix for outputs"""
```

**Parameters**
  - `batch_size` (int): The batch size.

**Returns**
  - `Tuple[np.ndarray, np.ndarray]`: A tuple containing the output covariance matrices.
