import numpy as np
import sys, os

from datetime import datetime
from matplotlib import pyplot

np.seterr(all='raise')

data_type = np.longdouble

def quadratic_error(result, expected):
    return np.sum(np.power(result - expected, 2), 1) / 2


def quadratic_error_back(result, expected):
    return result - expected


def fermi(integrated):
    return 1 / (1 + np.exp(-integrated))


def fermi_back(integrated, activated):
    return activated * (1 - activated)


def capped_fermi(integrated):
    result = np.copy(integrated)
    result[integrated > 100] = 1
    result[integrated < -100] = 0
    result[np.abs(integrated) <= 100] = fermi(result[np.abs(integrated) <= 100])
    return result


def identity(integrated):
    return integrated


def identity_back(integrated, activated):
    return np.ones(integrated.shape, dtype=data_type)


def tanh_back(integrated, activated):
    return 1 / np.cosh(integrated)**2

def with_bias(arr):
    return np.insert(arr, 0, 1, 1)

def average_quadratic_error(computed, targets):
    return np.sum(quadratic_error(computed, targets)) / computed.shape[0]


def gauss(integrated):
    return np.exp(-integrated ** 2 / 2)


class Layer:
    def propagate(self, inputs):
        pass

    def back_propagate(self, back_inputs):
        pass

    def train(self, learn_rate):
        pass

    def calculate(self, inputs):
        pass


class SumLayer(Layer):
    def __init__(self, in_size, out_size, activator, back_activator):
        self.weights = 2 * np.random.sample((in_size + 1, out_size)).astype(data_type) - 1
        self.activator = activator
        self.back_activator = back_activator
        self.last_delta = None

    def propagate(self, inputs):
        self.inputs = inputs
        self.integrated = with_bias(inputs) @ self.weights
        self.activated = self.activator(self.integrated)
        return self.activated

    def calculate(self, inputs):
        return self.activator(with_bias(inputs) @ self.weights)

    def back_propagate(self, back_inputs):
        self.back_inputs = back_inputs
        weights_without_bias = np.delete(self.weights, 0, 0)
        diffed = self.back_activator(self.integrated, self.activated)
        self.back_propagated = (diffed * back_inputs) @ weights_without_bias.T
        return self.back_propagated

    def learn(self, learn_rate, momentum=.0):
        delta = -learn_rate * (with_bias(self.inputs).T @ self.back_inputs)
        if momentum != .0 and self.last_delta is not None:
            delta += momentum * self.last_delta
        self.last_delta = delta
        self.weights += delta
        self.weights[np.isclose(self.weights, 0)] = 0


class DistanceLayer(Layer):
    def __init__(self, weights, activator):
        self.weights = weights
        self.activator = activator

    def propagate(self, inputs):
        return self.calculate(inputs)

    def calculate(self, inputs):
        s = self.weights.shape
        return np.array([np.linalg.norm(inp.reshape((s[0], 1)) @ np.ones((1, s[1])) - self.weights, axis=0) for inp in inputs])

    def back_propagate(self, back_inputs):
        pass

    def learn(self, learn_rate, momentum=.0):
        pass

class NeuralNet:
    def __init__(self, inputs):
        self.inputs = inputs
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def train_step(self, samples, targets, learn_rate, error_back = quadratic_error_back, momentum = .1, train_layer = None):
        try:
            calculation = samples
            for layer in self.layers:
                calculation = layer.propagate(calculation)

            calculation = error_back(calculation, targets)
            for layer in reversed(self.layers):
                calculation = layer.back_propagate(calculation)

            if train_layer is None:
                for layer in self.layers:
                    layer.learn(learn_rate, momentum=momentum)
            else:
                train_layer.learn(learn_rate, momentum=momentum)
        except:
            raise

    def calculate(self, input):
        calculation = input
        for layer in self.layers:
            calculation = layer.calculate(calculation)
        return calculation


def learn_full_batches(net, samples, targets, epochs, learn_rate, momentum=.0):
    print('training full batches with size {} in {} epochs with learn rate {}'.format(samples.shape[0], epochs, learn_rate))
    avg_error = [average_quadratic_error(net.calculate(samples), targets)]
    for i in range(epochs):
        net.train_step(samples, targets, learn_rate, momentum=momentum)
        avg_error.append(average_quadratic_error(net.calculate(samples), targets))
        if i % 1000 == 0:
            print('\r{:>3}%'.format(int(i * 100 / epochs)), end='')
            sys.stdout.flush()
    print('\r100%')
    return avg_error


def learn_random_order(net, samples, targets, epochs, learn_rate, momentum=.0):
    print('training {} samples in random order in {} epochs with learn rate {}'.format(samples.shape[0], epochs,
                                                                  learn_rate))
    avg_error = [average_quadratic_error(net.calculate(samples), targets)]
    for i in range(epochs):
        for j in np.random.permutation(samples.shape[0]):
            net.train_step(samples[j:j+1], targets[j:j+1], learn_rate, momentum=momentum)
        avg_error.append(average_quadratic_error(net.calculate(samples), targets))
        if i % 10 == 0:
            print('\r{:>3}%'.format(int(i * 100 / epochs)), end='')
            sys.stdout.flush()
    print('\r100%')
    return avg_error


def draw_error(avg_error, title, path = None, file = 'avg_error.png'):
    # show development of quadratic error
    generations = len(avg_error)
    gen_space = np.arange(0, generations, 1)
    pyplot.axes(xlim=(0, generations - 1), yscale='log', ylim=(0.01, 1))
    pyplot.plot(gen_space, avg_error)
    pyplot.suptitle('{}\ndevelopment of average quadratic error\nlog scale'.format(title))
    if path is not None:
        pyplot.savefig(path + file, bbox_inches='tight')
    else:
        pyplot.show()