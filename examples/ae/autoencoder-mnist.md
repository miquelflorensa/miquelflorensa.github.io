# Autoencoder MNIST

**Author:** [Miquel Florensa](https://www.linkedin.com/in/miquel-florensa/)  
**Date:** 2023/05/05  
**Description:** This example shows how to train an autoencoder to reconstruct the MNIST images.

<a href="https://github.com/miquelflorensa/miquelflorensa.github.io/blob/main/code/autoencoder_runner.py" class="github-link">
  <div class="github-icon-container">
    <img src="../../images/GitHub-Mark.png" alt="GitHub" height="32" width="64">
  </div>
  <div class="github-text-container">
    Github Source code
  </div>
</a>

---

## 1. Setup

We first import the required modules: the numpy library, the ImageViz, the autoencoder, the data loader and the encoder/decoder classes.

```python
import numpy as np
from visualizer import ImageViz

from python_examples.autoencoder import Autoencoder
from python_examples.data_loader import MnistDataloader
from python_examples.model import MnistDecoder, MnistEncoder
```

?>Notice that this modules are described [here](modules/modules.md) and the source code is in the *python_examples* directory, in case you have the modules in another directory you must change this paths.

## 2. Prepare the data

We define the number of epochs, some model properties and the paths to the data. Notice that the data is in ubyte format and divided in four files: the training images, the training labels, the test images and the test labels.

```python
# User-input
num_epochs = 10                  # row for 10 epochs
mu = np.array([0.1309])          # mean of each input
sigma = np.array([1])            # standard deviation of each input
img_size = np.array([1, 28, 28]) # size of image input
x_train_file = "./data/mnist/train-images-idx3-ubyte"
y_train_file = "./data/mnist/train-labels-idx1-ubyte"
x_test_file =  "./data/mnist/t10k-images-idx3-ubyte"
y_test_file =  "./data/mnist/t10k-labels-idx1-ubyte"
```

**You can find the used data in the [MNIST data](https://github.com/lhnguyen102/cuTAGI/tree/main/data/mnist) in the repository.*

## 3. Create the model

In this example we will create a model consisting in an encoder and a decoder that will allow us to reconstruct the original images. Find out more about the architecture in [Analytically Tractable Inference in Deep Neural Networks](https://arxiv.org/pdf/2103.05461.pdf).

```python
# Model
encoder_prop = MnistEncoder()
decoder_prop = MnistDecoder()
```

## 4. Load the data

The next step is to load the data. We will use the [MnistDataloader class](modules/data-loader?id=data-loader) to load the MNIST data and we will pass the batch size of the encoder and the data paths to the class.

```python
# Data loader
ae_data_loader = MnistDataloader(batch_size=encoder_prop.batch_size)
data_loader = ae_data_loader.process_data(x_train_file=x_train_file,
                                            y_train_file=y_train_file,
                                            x_test_file=x_test_file,
                                            y_test_file=y_test_file)
```

## 5. Create visualizer

In order to visualize the reconstruction of the images we can use the PredictionViz class.

```python
# Visualization
viz = ImageViz(task_name="autoencoder",
                data_name="mnist",
                mu=mu,
                sigma=sigma,
                img_size=img_size)
```

> Learn more about  PredictionViz class [here](https://github.com/lhnguyen102/cuTAGI/blob/main/visualizer.py).

## 6. Create the autoencoder object

Once we processed the data, we can create the autoencoder object. We will pass the number of epochs, the data loader, the network properties and vizualization object to the class.

```python
# Train and test
ae_task = Autoencoder(num_epochs=num_epochs,
                        data_loader=data_loader,
                        encoder_prop=encoder_prop,
                        decoder_prop=decoder_prop,
                        viz=viz)
```

> Find out more about the [Autoencoder class](modules/autoencoder.md).

## 7 Train and evaluate the model

Finally, we can train and evaluate the model. We will call the train and predict methods of the autoencoder object.

```python
ae_task.train()
ae_task.predict()
```

## 8. Results

If we created the visualizer object, we can visualize the results. The following figure shows the reconstructed images.

![autoencoder mnist](../../images/mnist_autoencoder_disp.png)
