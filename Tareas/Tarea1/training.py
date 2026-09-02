from network import Network
import numpy as np
from tensorflow.keras.datasets import mnist

if __name__=="__main__":
    # loading data base and normalizing
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    x_train = x_train.reshape(x_train.shape[0],784, 1)
    x_test = x_test.reshape(x_test.shape[0],784, 1)

    Red = Network(sizes=[784,120,10])
    Red.SGD(training_data=list(zip(x_train, y_train)),
     epochs=10, mini_batch_size=30,eta=0.01,
     test_data=list(zip(x_test, y_test)))
    