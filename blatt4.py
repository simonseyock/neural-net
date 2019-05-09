import numpy as np
from matplotlib import pyplot

from k_means import k_means, variance, create_clusters, choose_point
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

samples = np.vstack((class1.T, class2.T))
targets1 = np.ones((class1.shape[1], 1))
targets2 = -np.ones((class2.shape[1], 1))
targets = np.vstack((targets1, targets2))

# weights = np.stack((
#     np.random.sample(25) * (x[-1] - x[0]) + x[0],
#     np.random.sample(25) * (y[-1] - y[0]) + y[0]))
points = []
for i in range(25):
    points.append(choose_point(samples, points))
weights = np.array(points).T

clusters = create_clusters(samples, weights.T)
# variances = np.array([variance(samples, p) for p in weights.T])
variances = np.array([variance(clusters[i], weights.T[i]) for i in range(len(clusters))])

# means1, variances1 = k_means(class1.T, 12, init_k=4)
# means2, variances2 = k_means(class2.T, 13, init_k=5)
# weights = np.hstack((np.array(means1).T, np.array(means2).T))
# variances = np.hstack((variances1, variances2))
# variances = np.array([variance(samples, p) for p in weights.T])

# means = k_means(samples, 25, init_k=5)
# weights, variances = np.array(means[0]).T, np.array(means[1])
#
# variances = np.array(np.ones((1, weights.shape[1])))

net = NeuralNet(2)
net.add_layer(DistanceLayer(weights, variances, gauss))
net.add_layer(SumLayer(25, 1, np.tanh, tanh_back))

avg_error = learn_random_order(net, samples, targets, 10000, .0001)

draw_error(avg_error, 'avg error')

correctly_classified = np.sum(np.abs(net.calculate(samples) + targets) > 1)

result = net.calculate(grid)

pyplot.axes(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
cs = pyplot.contourf(x, y, reshape_grid(result, x.shape[0], y.shape[0]), levels = [-2, 0, 2])
# pyplot.clabel(cn, inline=1, fontsize=6)
pyplot.colorbar(cs)
pyplot.plot(class1[0], class1[1], '.', color='blue')
pyplot.plot(class2[0], class2[1], '.', color='red')
pyplot.plot(weights[0], weights[1], 'x')
pyplot.suptitle('correctly classified: {} / {}'.format(correctly_classified, samples.shape[0]))
pyplot.show()
