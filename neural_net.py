import numpy as np
import sys, os

from datetime import datetime
from matplotlib import pyplot

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


def with_bias(arr):
    return np.insert(arr, 0, 1, 1)


def target_function(x):
    return np.sin(x / 2) + np.cos(3 / (np.abs(x) + .1)) + .3 * x


def average_quadratic_error(computed, targets):
    return np.sum(quadratic_error(computed, targets)) / len(samples)


class Layer:
    def propagate(self, inputs):
        pass

    def back_propagate(self, back_inputs):
        pass

    def train(self, learn_rate):
        pass

    def calculate(self, inputs):
        pass


class SumLayer:
    def __init__(self, in_size, out_size, activator, back_activator):
        self.weights = 2 * np.random.sample((in_size + 1, out_size)).astype(data_type) - 1)
        self.activator = activator
        self.back_activator = back_activator
        self.last_delta = None

    def propagate(self, inputs):
        self.integrated = with_bias(inputs) @ self.weights
        self.activated = self.activator(self.integrated)
        return self.activated

    def calculate(self, inputs):
        return self.activator(with_bias(inputs) @ self.weights)

    def back_propagate(self, back_inputs):
        weights_without_bias = np.delete(self.weights, 0, 0)
        diffed = self.back_activator(self.integrated, self.activated)
        self.back_propagated = diffed * (weights_without_bias @ back_inputs.T).T
        return self.back_propagated

    def train(self, learn_rate, momentum=.0):
        delta = -learn_rate * with_bias(self.activated).T @ self.back_propagated
        if momentum != .0 and self.last_delta is not None:
            delta += momentum * self.last_delta
        self.last_delta = delta
        self.weights += delta


class NeuralNet:
    def __init__(self, inputs):
        self.inputs = inputs
        self.layers = []


    def add_layer(self, layer):
        self.layers.append(layer)


    def train_cycle(self, samples, targets, lern_rate, momentum=.1, train_layer=None):



    def train(self, samples, target, generations, learn_rate, layer=None, momentum=0.0):
        avg_error = []

        print('training {} generations with learn rate {}'.format(generations, learn_rate), end='')
        if momentum != 0.0:
            print(' and momentum factor {}'.format(momentum))
        else:
            print('')

        for i in range(generations):
            results = self.propagate(samples, target)
            computed = results[0][-1]
            avg_error.append(average_quadratic_error(computed, target))
            self.adjust_weights(learn_rate, results, layer, momentum)
            if i % 1000 == 0:
                print ('\r{:>3}%'.format(int(i * 100 / generations)), end='')
                sys.stdout.flush()

        computed = self.compute(samples)
        avg_error.append(average_quadratic_error(computed, target))
        print('\r100%')
        return avg_error


def draw_results(computed, samples, targets, title, path, file):
    pyplot.axes(xlim=(-10, 10))
    pyplot.plot(samples, computed, '.')
    pyplot.plot(samples, targets, '.')
    pyplot.suptitle('{}\naverage quadratic error: {}'.format(title, average_quadratic_error(computed, targets_vec)))
    if path is not None:
        pyplot.savefig(path + file, bbox_inches='tight')
    else:
        pyplot.show()


def draw_error(avg_error, title, path, file):
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


def save_weights(net, path, file):
    if path is not None:
        np.save(path + file, net.weights)


output_folder = './out'
nets = 1

for i in range(nets):
    # stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # path = '{}/{}/'.format(output_folder, stamp)
    # os.makedirs(path)
    path = None

    # create samples
    samples = np.linspace(-10, 10, 1001, dtype=data_type)
    targets = target_function(samples)

    samples_vec = np.array([[e] for e in samples], dtype=data_type)
    targets_vec = np.array([[e] for e in targets], dtype=data_type)

    # create net
    net = NeuralNet(1)
    # net.add_layer(100, fermi, fermi_back)
    # net.add_layer(100, fermi, fermi_back)
    net.add_layer(20, capped_fermi, fermi_back)
    net.add_layer(1, identity, identity_back)

    generations = 10000
    learn_rate = .001
    momentum = 0

    save_weights(net, path, 'weights_1.txt')

    # before training
    computed = net.compute(samples_vec)
    draw_results(computed, samples, targets, 'Before training', path, 'graph_1.png')

    # save weights
    stored_weights = np.copy(net.weights)

    # training only second layer
    avg_error = net.train(samples_vec, targets_vec, generations, learn_rate, 1)
    draw_error(avg_error, 'After training only second layer', path, 'graph_2.png')
    computed = net.compute(samples_vec)
    draw_results(computed, samples, targets, 'After training only second layer', path, 'graph_3.png')

    save_weights(net, path, 'weights_2.txt')

    # reset net
    net.weights = stored_weights

    if path is not None:
        with open(path + 'meta.txt', 'w') as file:
            file.write('layers: {}\n'.format('-'.join([str(net.weights[0].shape[0] - 1)] + [str(w.shape[1]) for w in net.weights])))
            file.write('generations: {}\n'.format(generations))
            file.write('learn_rate: {}\n'.format(learn_rate))
            file.write('momentum: {}\n'.format(momentum))
            if net.activators[0] == capped_fermi:
                file.write('using capped fermi\n')

    # training both layers
    avg_error = net.train(samples_vec, targets_vec, generations, learn_rate, momentum=0.5)
    draw_error(avg_error, 'After training both layers', path, 'graph_4.png')
    computed = net.compute(samples_vec)
    draw_results(computed, samples, targets, 'After training both layers', path, 'graph_5.png')

    save_weights(net, path, 'weights_3.txt')
