import random
import numpy as np

def k_means(points, k):
    midpoints = [weighted_middle(points)]
    if k % 2 != 1:
        raise AttributeError('this implementation needs an uneven number of points')
    while len(midpoints) < k:
        clusters = create_clusters(points, midpoints)
        midpoints = [weighted_middle(cluster) for cluster in clusters]
        _, greatest_error = max(clusters, lambda c, i: error(c, midpoints[i]))
        _, most_elements = max(clusters, lambda c, i: len(c))
        midpoints.append(choose_point(clusters[greatest_error], midpoints))
        midpoints.append(choose_point(clusters[most_elements], midpoints))
    last_midpoints = midpoints
    clusters = create_clusters(points, midpoints)
    midpoints = [weighted_middle(cluster) for cluster in clusters]


def choose_point(points, not_in):
    new = random.choice(points)
    while new in not_in:
        new = random.choice(points)
    return new


def error(cluster, midpoint):
    return np.linalg.norm(np.array(cluster) - midpoint)**2


def max(list, func):
    max = 0
    max_value = func(list[0])
    for i in range(1, len(list)):
        if func(list[i]) > max_value:
            max = i
            max_value = func(list[i], i)
    return max_value, max


def create_clusters(points, midpoints):
    clusters = [[]] * len(midpoints)
    for p in points:
        _, nearest = max(midpoints, lambda mp, i: np.linalg.norm(p, mp))
        clusters[nearest].append(p)
    return clusters


def weighted_middle(points):
    return np.sum(points, axis=0) / points.shape[0]
