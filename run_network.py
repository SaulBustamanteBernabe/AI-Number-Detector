import cv2
import gzip
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle

from classes.Network import Network


def load_data():
    mnist = gzip.open('mnist.pkl.gz', 'rb')
    training_data, classification_data, test_data = pickle.load(mnist, encoding='latin1')
    mnist.close()

    return (training_data, classification_data, test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def wrap_data():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def display_images(data):
    for i in range(10):
        for p in data[0][i]:
            print(p)
        image = data[0][i].reshape((28, 28))
        label = data[1][i]
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.show()


if __name__ == "__main__":
    training_data, validation_data, test_data = wrap_data()
    net = Network([784, 30, 10])

    if os.path.isfile("weights.pkl") and os.path.isfile("biases.pkl"):
        with open('weights.pkl', 'rb') as w:
            net.weights = pickle.load(w)
        with open('biases.pkl', 'rb') as b:
            net.biases = pickle.load(b)
    else:
        net.SGD(list(training_data), 30, 10, 3.0, test_data=list(test_data))
        with open('weights.pkl', 'wb') as w:
            pickle.dump(net.weights, w)
        with open('biases.pkl', 'wb') as b:
            pickle.dump(net.biases, b)

    img_num = cv2.imread('Numero.png', cv2.IMREAD_GRAYSCALE)
    array_num = (np.reshape(cv2.bitwise_not(img_num), (784, 1))/255)
    res = net.feedforward(array_num)
    print(np.argmax(res))
