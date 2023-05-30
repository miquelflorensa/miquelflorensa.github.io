# TimeSeriesForecaster

The `TimeSeriesForecaster` class is responsible for time series forecasting using TAGI.

## Constructor

### `__init__(self, num_epochs: int, data_loader: dict, net_prop: NetProp, param: Union[Param, None] = None, viz: Union[PredictionViz, None] = None, dtype=np.float32) -> None`

Constructor method for the `TimeSeriesForecaster` class.

- Parameters:
  - `num_epochs` (int): Number of epochs for training.
  - `data_loader` (dict): Data loader dictionary containing training and testing data.
  - `net_prop` (NetProp): Network properties.
  - `param` (Union[Param, None], optional): Network parameters. Defaults to None.
  - `viz` (Union[PredictionViz, None], optional): Visualization object. Defaults to None.
  - `dtype` (np.float32, optional): Data type. Defaults to np.float32.

- Returns: None

## Methods

### `train(self) -> None`

- Description: Train LSTM network.

- Parameters: None
- Returns: None

### `predict(self) -> None`

- Description: Make prediction for time series using TAGI.

- Parameters: None
- Returns: None

### `init_inputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]`

- Description: Initialize the covariance matrix for inputs.

- Parameters:
  - `batch_size` (int): Batch size.

- Returns:
  - `Tuple[np.ndarray, np.ndarray]`: Initialized inputs.

### `init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]`

- Description: Initialize the covariance matrix for outputs.

- Parameters:
  - `batch_size` (int): Batch size.

- Returns:
  - `Tuple[np.ndarray, np.ndarray]`: Initialized outputs.
