<!-------------------------------------------------------------------
File:         tutorial.md
Description:  FNN tutorial with 1D data
Authors:      Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet
Created:      March 04, 2023
Updated:      May 29, 2023
Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com & james.goulet@polymtl.ca
Copyright (c) 2023 Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
-------------------------------------------------------------------->

# data_loader.py

<a href="https://github.com/lhnguyen102/cuTAGI/blob/main/python_examples/data_loader.py" class="github-link">
  <div class="github-icon-container">
    <img src="../images/GitHub-Mark.png" alt="GitHub" height="32" width="64">
  </div>
  <div class="github-text-container">
    Github Source code
  </div>
</a>

# The DataloaderBase class

This class represents a template for a data loader.

## Atributes

- `normalizer`: An instance from the [Normalizer](api/utils?id=the-normalizer-class) class.

## *constructor* method

> Constructor for the DataloaderBase class.

```python
def __init__(self, batch_size: int) -> None:
```

**Parameters**
- `batch_size`: An integer representing the batch size.

## *process_data* method

```python
@abstractmethod
def process_data(self) -> dict:
    """Abstract method for processing the data"""
```

**Returns**
- `dict`: A dictionary containing the processed data.

## *create_data_loader* method

```python
def create_data_loader(self, raw_input: np.ndarray, raw_output: np.ndarray) -> list:
    """Create dataloader based on batch size"""
```

**Parameters**
- `raw_input`: Raw input data as a NumPy array.
- `raw_output`: Raw output data as a NumPy array.

**Returns**
- `list`: A list of tuples representing the input-output pairs in each batch.

## *split_data* method

```python
@staticmethod
def split_data(data: int, test_ratio: float = 0.2, val_ratio: float = 0.0) -> dict:
    """Split data into training, validation, and test sets"""
```

**Parameters**
- `data`: Input data as a NumPy array.
- `test_ratio`: Optional. Float representing the ratio of test data (default: 0.2).
- `val_ratio`: Optional. Float representing the ratio of validation data (default: 0.0).

**Returns**
- `dict`: A dictionary containing the split data sets.

## *load_data_from_csv* method

```python
@staticmethod
def load_data_from_csv(data_file: str) -> pd.DataFrame:
    """Load data from a CSV file"""
```

**Parameters**
- `data_file`: Path to the CSV file.

**Returns**
- `pd.DataFrame`: The loaded data as a Pandas DataFrame.

## *split_evenly* method

```python
@staticmethod
def split_evenly(num_data, chunk_size: int):
    """Split data evenly"""
```

**Parameters**
- `num_data`: The number of data points.
- `chunk_size`: The size of each chunk.

**Returns**
- `np.ndarray`: An array of indices representing the split data.

## *split_reminder* method

```python
@staticmethod
def split_reminder(num_data: int, chunk_size: int):
    """Pad the reminder"""
```

**Parameters**
- `num_data`: The number of data points.
- `chunk_size`: The size of each chunk.

**Returns**
- `np.ndarray`: An array of indices representing the split data.

---

<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->

# The RegressionDataLoader class

A class for loading and formatting data that is fed to a neural network. This class inherits from the DataloaderBase class.

## *constructor* method

> Constructor for the RegressionDataLoader class.

```python
def __init__(self, batch_size: int, num_inputs: int, num_outputs: int) -> None:
```

**Parameters**
- `batch_size`: An integer representing the batch size.
- `num_inputs`: An integer representing the number of input features.
- `num_outputs`: An integer representing the number of output features.

## *process_data* method

```python
def process_data(self, x_train_file: str, y_train_file: str, 
                 x_test_file: str, y_test_file: str) -> dict:
    """Process data from the csv file"""
```

**Parameters**
- `x_train_file`: A string representing the file path of the input training data in CSV format.
- `y_train_file`: A string representing the file path of the output training data in CSV format.
- `x_test_file`: A string representing the file path of the input testing data in CSV format.
- `y_test_file`: A string representing the file path of the output testing data in CSV format.

**Returns**
- `dict`: A dictionary containing the processed data and normalization parameters.

<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->

# MnistDataloader class

Data loader for mnist dataset.

## *process_data* method

```python
def process_data(self, x_train_file: str, y_train_file: str,
                 x_test_file: str, y_test_file: str) -> dict:
    """Process mnist images"""
```

**Parameters**
- `x_train_file`: Path to the file containing mnist training images.
- `y_train_file`: Path to the file containing mnist training labels.
- `x_test_file`: Path to the file containing mnist test images.
- `y_test_file`: Path to the file containing mnist test labels.

**Returns**
- `dict`: A dictionary containing the processed data.

<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->

# The TimeSeriesDataloader class

Data loader for time series.

## *constructor* method

> Constructor for the TimeSeriesDataloader class.

```python
def __init__(self, batch_size: int, output_col: np.ndarray,
             input_seq_len: int, output_seq_len: int, num_features: int,
             stride: int) -> None:
```

**Parameters**
- `batch_size`: An integer representing the batch size.
- `output_col`: A NumPy array representing the output column.
- `input_seq_len`: An integer representing the length of the input sequence.
- `output_seq_len`: An integer representing the length of the output sequence.
- `num_features`: An integer representing the number of features.
- `stride`: An integer representing the stride.

## *process_data* method

```python
def process_data(self, x_train_file: str, datetime_train_file: str,
                 x_test_file: str, datetime_test_file: str) -> dict:
    """Process time series"""
```

**Parameters**
- `x_train_file`: A string representing the file path for training input data.
- `datetime_train_file`: A string representing the file path for training datetime data.
- `x_test_file`: A string representing the file path for testing input data.
- `datetime_test_file`: A string representing the file path for testing datetime data.

**Returns**
- `dict`: A dictionary containing the processed data.

<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->
<!--########################################################################-->

# ClassificationDataloader class

Data loader for csv dataset for classification.

## *constructor* method

> Constructor for the ClassificationDataloader class.

```python
def __init__(self, batch_size: int) -> None:
```

**Parameters**
- `batch_size`: An integer representing the batch size for the data loader.

## *process_data* method

```python
def process_data(self, x_train_file: str, y_train_file: str,
                 x_test_file: str, y_test_file: str) -> dict:
    """Process data from the csv file"""
```

**Parameters**
- `x_train_file`: File path for the training images CSV file.
- `y_train_file`: File path for the training labels CSV file.
- `x_test_file`: File path for the test images CSV file.
- `y_test_file`: File path for the test labels CSV file.

**Returns**
- `dict`: A dictionary containing the processed data.
