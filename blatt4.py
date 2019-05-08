import numpy as np
from matplotlib import pyplot

from neural_net import *

def to_polar(coordinates):
    return np.stack((
        np.arctan2(coordinates[1], coordinates[0]),
        np.linalg.norm(coordinates, axis=0)))

path = None
polar = True

u = np.arange(1, 201)

class1 = np.stack((2 + np.sin(.2 * u + 8) * np.sqrt(u + 10), -1 + np.cos(.2 * u + 8) * np.sqrt(u + 10)))
class2 = np.stack((2 + np.sin(.2 * u - 8) * np.sqrt(u + 10), -1 + np.cos(.2 * u - 8) * np.sqrt(u + 10)))
if polar:
    class1 = to_polar(class1)
    class2 = to_polar(class2)

if not polar:
    x = np.arange(-16, 16.1, .1)
    y = np.arange(-16, 16.1, .1)
else:
    x = np.arange(-np.floor(10 * np.pi) / 10, np.floor(10 * np.pi) / 10, .1)
    y = np.arange(0, np.sqrt(16**2), .1)

def make_grid(x, y):
    grid = np.zeros((x.shape[0] * y.shape[0], 2))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            grid[i * y.shape[0] + j] = (x[i], y[j])
    return grid

def reshape_grid(grid, xshape, yshape):
    reshaped = np.zeros((yshape, xshape))
    for i in range(xshape):
        for j in range(yshape):
            reshaped[j, i] = grid[i * yshape + j][0]
    return reshaped

grid = make_grid(x, y)

random_weights = np.stack((
    np.random.sample(25) * (x[-1] - x[0]) + x[0],
    np.random.sample(25) * (y[-1] - y[0]) + y[0]))

net = NeuralNet(2)
net.add_layer(DistanceLayer(random_weights, gauss))
net.add_layer(SumLayer(25, 1, np.tanh, tanh_back))

samples = np.vstack((class1.T, class2.T))
targets1 = np.ones((class1.shape[1], 1))
targets2 = -np.ones((class2.shape[1], 1))
targets = np.vstack((targets1, targets2))

avg_error = learn_random_order(net, samples, targets, 1000, .0001)

draw_error(avg_error, 'avg error')

result = net.calculate(grid)

pyplot.axes(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
cs = pyplot.contourf(x, y, reshape_grid(result, x.shape[0], y.shape[0]))
# pyplot.clabel(cn, inline=1, fontsize=6)
pyplot.colorbar(cs)
pyplot.plot(class1[0], class1[1], '.', color='blue')
pyplot.plot(class2[0], class2[1], '.', color='red')
pyplot.plot(random_weights[0], random_weights[1], 'x')
pyplot.show()
