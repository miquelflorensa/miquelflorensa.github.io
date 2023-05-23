from python_examples.data_loader import RegressionDataLoader
from python_examples.regression import Regression
from pytagi import NetProp

class HeterosUCIMLP(NetProp):
    """Multi-layer preceptron for regression task where the
    output's noise varies overtime"""

    def __init__(self) -> None:
        super().__init__()
        self.layers: list = [1, 1, 1, 1]
        self.nodes: list = [13, 50, 50, 2]  # output layer = [mean, std]
        self.activations: list = [0, 4, 4, 0]
        self.batch_size: int = 10
        self.sigma_v: float = 2
        self.sigma_v_min: float = 0.3
        self.noise_type: str = "heteros"
        self.noise_gain: float = 1.0
        self.init_method: str = "He"
        self.device: str = "cpu"

def main():
    """Training and testing API"""
    # User-input
    num_inputs = 1
    num_outputs = 1
    num_epochs = 50
    x_train_file = "./data/toy_example/x_train_1D_noise_inference.csv"
    y_train_file = "./data/toy_example/y_train_1D_noise_inference.csv"
    x_test_file = "./data/toy_example/x_test_1D_noise_inference.csv"
    y_test_file = "./data/toy_example/y_test_1D_noise_inference.csv"

    # Model
    net_prop = HeterosUCIMLP()

    # Data loader
    reg_data_loader = RegressionDataLoader(num_inputs=num_inputs,
                                           num_outputs=num_outputs,
                                           batch_size=net_prop.batch_size)
    data_loader = reg_data_loader.process_data(x_train_file=x_train_file,
                                               y_train_file=y_train_file,
                                               x_test_file=x_test_file,
                                               y_test_file=y_test_file)

    # Train and test
    reg_task = Regression(num_epochs=num_epochs,
                          data_loader=data_loader,
                          net_prop=net_prop)
    reg_task.train()
    reg_task.predict()


if __name__ == "__main__":
    main()
