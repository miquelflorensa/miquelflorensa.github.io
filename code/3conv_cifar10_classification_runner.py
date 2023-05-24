from python_examples.classification import Classifier
from python_examples.data_loader import ClassificationDataloader
from python_examples.model import ConvCifarMLP


def main():
    """Training and testing API"""
    # User-input
    num_epochs = 50
    x_train_file = "./data/cifar/x_train.csv"
    y_train_file = "./data/cifar/y_train.csv"
    x_test_file = "./data/cifar/x_test.csv"
    y_test_file = "./data/cifar/y_test.csv"

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
