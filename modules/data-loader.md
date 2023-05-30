<!-------------------------------------------------------------------
File:         tutorial.md
Description:  FNN tutorial with 1D data
Authors:      Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet
Created:      March 04, 2023
Updated:      May 29, 2023
Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com & james.goulet@polymtl.ca
Copyright (c) 2023 Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
-------------------------------------------------------------------->

# Data loader

This file contains classes and functions for preparing data for neural networks.

## DataloaderBase (abstract class)
```python
class DataloaderBase(ABC):
    """Dataloader template"""
```

> This is an abstract base class that serves as a template for other dataloader classes.
> 
> It provides common methods and functionality for data loading and processing.
>
> Subclasses are expected to implement the `process_data` method according to their specific data requirements.

### Methods
- `__init__(self, batch_size: int) -> None`: Initializes the DataloaderBase object with the specified batch size.
- `process_data(self) -> dict`: Abstract method to be implemented by subclasses for processing data.
- `create_data_loader(self, raw_input: np.ndarray, raw_output: np.ndarray) -> list`: Creates a data loader based on the batch size.
- `split_data(data: int, test_ratio: float = 0.2, val_ratio: float = 0.0) -> dict`: Splits the data into training, validation, and test sets.
- `load_data_from_csv(data_file: str) -> pd.DataFrame`: Loads data from a CSV file.
- `split_evenly(num_data, chunk_size: int)`: Splits the data evenly.
- `split_reminder(num_data: int, chunk_size: int)`: Pads the remaining data.

## RegressionDataLoader (subclass of DataloaderBase)
```python
class RegressionDataLoader(DataloaderBase):
    """Load and format data that are fed to the neural network.
     The user must provide the input and output data file in CSV format"""
```

> This class is responsible for loading and formatting data for regression tasks.
>
> It takes input and output data files in CSV format and processes them for training and testing.

### Methods
- `__init__(self, batch_size: int, num_inputs: int, num_outputs: int) -> None`: Initializes the RegressionDataLoader object with the specified batch size, number of inputs, and number of outputs.
- `process_data(self, x_train_file: str, y_train_file: str, x_test_file: str, y_test_file: str) -> dict`: Processes data from CSV files for regression tasks.

## MnistDataloader (subclass of DataloaderBase)
```python
class MnistDataloader(DataloaderBase):
    """Data loader for MNIST dataset"""
```
> This class is a data loader specifically designed for the MNIST dataset, which consists of handwritten digit images.
>
> It loads and preprocesses the MNIST image data for training and testing.

### Methods
- `process_data(self, x_train_file: str, y_train_file: str, x_test_file: str, y_test_file: str) -> dict`: Processes MNIST images data.

## MnistOneHotDataloader (subclass of DataloaderBase)
```python
class MnistOneHotDataloader(DataloaderBase):
    """Data loader for MNIST dataset"""
```
> This class is similar to the `MnistDataloader` but additionally performs one-hot encoding on the labels.
>
> It converts the categorical label values into binary vectors to be used in classification tasks.

### Methods
- `process_data(self, x_train_file: str, y_train_file: str, x_test_file: str, y_test_file: str) -> dict`: Processes MNIST images data and uses one-hot encoding for labels.

## TimeSeriesDataloader (subclass of DataloaderBase)
```python
class TimeSeriesDataloader(DataloaderBase):
    """Data loader for time series"""
```
> This class is designed for loading and processing time series data.
>
> It takes input data files containing time series sequences and associated timestamps.
>
> The class organizes the data into input-output pairs based on specified sequence lengths and stride values.

### Methods
- `__init__(self, batch_size: int, output_col: np.ndarray, input_seq_len: int, output_seq_len: int, num_features: int, stride: int) -> None`: Initializes the TimeSeriesDataloader object with the specified parameters.
- `process_data(self, x_train_file: str, datetime_train_file: str, x_test_file: str, datetime_test_file: str) -> dict`: Processes time series data.

## ClassificationDataloader (subclass of DataloaderBase)
```python
class ClassificationDataloader(DataloaderBase):
    """Data loader for CSV dataset for classification"""
```
> This class is responsible for loading and formatting data for classification tasks.
>
> It takes input and output data files in CSV format and processes them for training and testing.

### Methods
- `__init__(self, batch_size: int) -> None`: Initializes the ClassificationDataloader object with the specified batch size.
- `process_data(self, x_train_file: str, y_train_file: str, x_test_file: str, y_test_file: str) -> dict`: Processes data from CSV files for classification tasks.


