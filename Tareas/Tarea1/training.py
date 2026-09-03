from network import Network
import numpy as np
from tensorflow.keras.datasets import mnist

if __name__=="__main__":
    # loading data base and normalizing
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    num_classes = 10
    # In the funtion cost_derivative we can see that y_train needs to be in one-hot
    # but in the funtion evaluate y_test needs to be a constant
    y_train_one_hot = np.eye(num_classes, dtype=np.float32)[y_train]
    y_train_one_hot = y_train_one_hot.reshape(y_train_one_hot.shape[0],10,1)
    y_test_one_hot = np.eye(num_classes, dtype=np.float32)[y_test]
    y_test_one_hot = y_test_one_hot.reshape(y_test_one_hot.shape[0],10,1)

    x_train = x_train.reshape(x_train.shape[0],784, 1)
    x_test = x_test.reshape(x_test.shape[0],784, 1)

    Red = Network(sizes=[784,240,120,10])
    Red.SGD(training_data=list(zip(x_train, y_train_one_hot)),
     epochs=30, mini_batch_size=120,eta=0.01,
     test_data=list(zip(x_test, y_test_one_hot)))
    