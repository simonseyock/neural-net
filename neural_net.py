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


class NeuralNet:
    def __init__(self, inputs):
        self.inputs = inputs
        self.weights = []
        self.activators = []
        self.back_activators = []
        self.last_delta = []


    def add_layer(self, size, activator, back_activator):
        if len(self.weights) == 0:
            self.weights.append(2 * np.random.sample((self.inputs + 1, size)).astype(data_type) - 1)
        else:
            last = self.weights[-1]
            self.weights.append(2 * np.random.sample((last.shape[1] + 1, size)).astype(data_type) - 1)
        self.activators.append(activator)
        self.back_activators.append(back_activator)
        self.last_delta.append(np.zeros(self.weights[-1].shape, dtype=data_type))


    def propagate(self, input, target, error_back=quadratic_error_back):
        layers = len(self.weights)

        integrated = []
        activated = [input]
        back_propagated = [None] * layers

        for i in range(layers):
            # integration
            integrated.append(with_bias(activated[i]) @ self.weights[i])

            # activation
            activated.append(self.activators[i](integrated[i]))

        # back propagation
        back_propagated[layers - 1] = error_back(activated[-1], target)

        for i in range(layers - 2, -1, -1):
            weights_without_bias = np.delete(self.weights[i + 1], 0, 0)
            diffed = self.back_activators[i](integrated[i], activated[i + 1])
            back_propagated[i] = diffed * (weights_without_bias @ back_propagated[i + 1].T).T

        return activated, back_propagated


    def adjust_weights(self, learn_rate, results, layer=None, momentum=0.):
        if layer is None:
            for i in range(len(self.weights)):
                self.adjust_weights(learn_rate, results, i, momentum)
        else:
            activated, back_propagated = results
            delta = -learn_rate * with_bias(activated[layer]).T @ back_propagated[layer] + momentum * self.last_delta[layer]
            self.last_delta[layer] = delta
            self.weights[layer] += delta


    def compute(self, input):
        values = input
        for i in range(len(self.weights)):
            # integration
            values = with_bias(values) @ self.weights[i]

            # activation
            values = self.activators[i](values)

        return values


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