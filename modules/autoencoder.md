# Autoencoder

The `Autoencoder` class represents an autoencoder task using TAGI.

## Constructor

### `__init__(self, num_epochs: int, data_loader: dict, encoder_prop: NetProp, decoder_prop: NetProp, encoder_param: Union[Param, None] = None, decoder_param: Union[Param, None] = None, viz: Union[ImageViz, None] = None, dtype=np.float32) -> None`

Constructor method for the `Autoencoder` class.

- Parameters:
  - `num_epochs` (int): Number of epochs for training.
  - `data_loader` (dict): Data loader dictionary containing training and testing data.
  - `encoder_prop` (NetProp): Network properties for the encoder.
  - `decoder_prop` (NetProp): Network properties for the decoder.
  - `encoder_param` (Union[Param, None], optional): Network parameters for the encoder. Defaults to None.
  - `decoder_param` (Union[Param, None], optional): Network parameters for the decoder. Defaults to None.
  - `viz` (Union[ImageViz, None], optional): Image visualization object. Defaults to None.
  - `dtype` (np.float32, optional): Data type. Defaults to np.float32.

- Returns: None

## Methods

### `train(self) -> None`

Train the encoder and decoder networks.

- Parameters: None
- Returns: None

### `predict(self) -> None`

Generate images using the autoencoder.

- Parameters: None
- Returns: None

### `init_inputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]`

Initialize the covariance matrix for inputs.

- Parameters:
  - `batch_size` (int): Batch size.

- Returns:
  - `Tuple[np.ndarray, np.ndarray]`: A tuple containing the input covariance matrices.

### `init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]`

Initialize the covariance matrix for outputs.

- Parameters:
  - `batch_size` (int): Batch size.

- Returns:
  - `Tuple[np.ndarray, np.ndarray]`: A tuple containing the output covariance matrices.
