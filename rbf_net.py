import numpy as np
from matplotlib import pyplot

def draw_target(class1, class2, title, path, file):
    # pyplot.axes(xlim=(-10, 10))
    pyplot.plot(class1[0], class1[1], '.')
    pyplot.plot(class2[0], class2[1], '.')
    # pyplot.suptitle('{}\naverage quadratic error: {}'.format(title, average_quadratic_error(computed, targets_vec)))
    if path is not None:
        pyplot.savefig(path + file, bbox_inches='tight')
    else:
        pyplot.show()

def to_polar(coordinates):
    return np.stack((
        np.arctan2(coordinates[1], coordinates[0]),
        np.linalg.norm(coordinates, axis=0)))

path = None

u = np.arange(1, 201)

class1 = np.stack((2 + np.sin(.2 * u + 8) * np.sqrt(u + 10), -1 + np.cos(.2 * u + 8) * np.sqrt(u + 10)))
class2 = np.stack((2 + np.sin(.2 * u - 8) * np.sqrt(u + 10), -1 + np.cos(.2 * u - 8) * np.sqrt(u + 10)))

draw_target(class1, class2, 'cartesian coordinates', path, 'targets.png')

# draw_target(to_polar(class1), to_polar(class2), 'cartesian coordinates', path, 'targets.png')
