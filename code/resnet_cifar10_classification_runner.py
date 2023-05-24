from python_examples.classification import Classifier
from python_examples.data_loader import ClassificationDataloader
from pytagi import NetProp

class ConvCifarMLP(NetProp):
    """Multi-layer perceptron for cifar classificaiton."""

    def __init__(self) -> None:
        super().__init__()
        #------------------- #Input-----------------#Stage1-------------------------#Stage2-------------------------#Stage3-------------------------#Stage4-------------------------#Output---
        self.layers =  	    [2,     2,      4,      2,      4,      2,      4,      2,      4,      2,      4,      1,      1]
        self.nodes = 	    [3072,  0,      0,      0,      0,      0,      0,      0,      0,      0,      0,    256,     11]
        self.kernels = 	    [7,     1,      3,      1,      3,      1,      3,      1,      3,      1,      3,      1,      1]
        self.strides = 	    [1,     0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      0]
        self.widths =  	    [128, 128,     64,     64,     32,     32,     16,     16,      8,      8,      4,      1,      1]
        self.heights = 	    [128, 128,     64,     64,     32,     32,     16,     16,      8,      8,      4,      1,      1]
        self.filters =      [3,    32,     32,     32,     32,     64,     64,    128,    128,    256,    256,      1,      1]
        self.pads =         [0,     1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0]
        self.pad_types =    [0,     1,      0,      1,      0,      1,      0,      1,      0,      1,      0,      1,      0]
        self.activations =  [0,     4,      0,      4,      0,      4,      0,      4,      0,      4,      0,      4,     12]
        self.shortcuts =    [-1,    3,     -1,      6,     -1,     -1,     -1,      7,     -1,     -1,     -1,     -1,     -1]
        self.batch_size = 16
        self.sigma_v = 1
        self.sigma_v_min = 0.2
        self.decay_factor_sigma_v = 0.975
        self.is_idx_ud = True
        self.multithreading = True
        self.init_method: str = "He"
        self.device = "cuda"

def main():
    """Training and testing API"""
    # User-input
    num_epochs = 1
    x_train_file = "./data/cifar10/x_train.csv"
    y_train_file = "./data/cifar10/y_train.csv"
    x_test_file = "./data/cifar10/x_test.csv"
    y_test_file = "./data/cifar10/y_test.csv"

    # Model
    net_prop = ConvCifarMLP()

    # Data loader
    reg_data_loader = ClassificationDataloader(batch_size=net_prop.batch_size)
    data_loader = reg_data_loader.process_data(x_train_file=x_train_file,
                                               y_train_file=y_train_file,
                                               x_test_file=x_test_file,
                                               y_test_file=y_test_file)

    # Train and test
    clas_task = Classifier(num_epochs=num_epochs,
                          data_loader=data_loader,
                          net_prop=net_prop,
                          num_classes=10)
    clas_task.train()
    clas_task.predict()


if __name__ == "__main__":
    main()