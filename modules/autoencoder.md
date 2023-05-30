# Autoencoder class

The `Autoencoder` class is responsible for performing the autoencoder task using the TAGI algorithm.

<a href="https://github.com/miquelflorensa/cuTAGI/blob/main/python_examples/autoencoder.py" class="github-link">
  <div class="github-icon-container">
    <img src="../images/GitHub-Mark.png" alt="GitHub" height="32" width="64">
  </div>
  <div class="github-text-container">
    GitHub Source code
  </div>
</a>

## Attributes

- `utils`: An instance of the `Utils` class.
- `num_epochs`: The number of epochs for training.
- `data_loader`: A dictionary containing the data loader.
- `encoder_prop`: An instance of the `NetProp` class representing the properties of the encoder network.
- `decoder_prop`: An instance of the `NetProp` class representing the properties of the decoder network.
- `encoder`: An instance of the `TagiNetwork` class representing the encoder network.
- `decoder`: An instance of the `TagiNetwork` class representing the decoder network.
- `viz`: An optional instance of the `ImageViz` class for visualization.
- `dtype`: The data type (default: `np.float32`).

## *constructor* method

> Constructor for the Autoencoder class.

```python
def __init__(
    self,
    num_epochs: int,
    data_loader: dict,
    encoder_prop: NetProp,
    decoder_prop: NetProp,
    encoder_param: Union[Param, None] = None,
    decoder_param: Union[Param, None] = None,
    viz: Union[ImageViz, None] = None,
    dtype=np.float32
) -> None:
```

**Parameters**
- `num_epochs`: An integer representing the number of epochs for training.
- `data_loader`: A dictionary containing the data loader.
- `encoder_prop`: An instance of the `NetProp` class representing the properties of the encoder network.
- `decoder_prop`: An instance of the `NetProp` class representing the properties of the decoder network.
- `encoder_param`: An optional instance of the `Param` class representing the parameters of the encoder network (default: `None`).
- `decoder_param`: An optional instance of the `Param` class representing the parameters of the decoder network (default: `None`).
- `viz`: An optional instance of the `ImageViz` class for visualization (default: `None`).
- `dtype`: The data type (default: `np.float32`).

## *train* method

```python
def train(self) -> None:
    """Train encoder and decoder"""
```

Trains the encoder and decoder networks using the TAGI algorithm. It performs the following steps:
1. Initializes inputs and outputs.
2. Performs training iterations for each epoch.
3. Updates the network parameters and hidden states.
4. Computes the loss and displays the progress.
5. Calls the `predict` method.

## *predict* method

```python
def predict(self) -> None:
    """Generate images"""
```

Generates images using the trained encoder and decoder networks. It performs the following steps:
1. Initializes inputs.
2. Makes predictions using the encoder and decoder networks.
3. Retrieves the generated images.
4. Performs visualization if the `viz` attribute is not `None`.

## *init_inputs* method

```python
def init_inputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize the covariance matrix for inputs"""
```

Initializes the covariance matrix for the inputs. It returns the initialized covariance matrices `Sx_batch` and `Sx_f_batch`.



**Parameters**
- `batch_size`: An integer representing the batch size.

**Returns**
- A tuple containing the initialized covariance matrices `Sx_batch` and `Sx_f_batch`.

## *init_outputs* method

```python
def init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize the covariance matrix for outputs"""
```

Initializes the covariance matrix for the outputs. It returns the initialized covariance matrices `V_batch` and `ud_idx_batch`.

**Parameters**
- `batch_size`: An integer representing the batch size.

**Returns**
- A tuple containing the initialized covariance matrices `V_batch` and `ud_idx_batch`.