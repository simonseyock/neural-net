import random
import numpy as np

def k_means(points, k, init_k=1):
    if (k - init_k) % 2 != 0:
        raise AttributeError('k - init_k needs to be even')
    midpoints = []
    for i in range(init_k):
        midpoints.append(choose_point(points, midpoints))
    while len(midpoints) < k:
        clusters = create_clusters(points, midpoints)
        midpoints = [weighted_middle(cluster) for cluster in clusters]
        _, greatest_error = find(clusters, lambda c, i: variance(c, midpoints[i]), max_comp)
        _, most_elements = find(clusters, lambda c, i: len(c), max_comp)
        midpoints.append(choose_point(clusters[greatest_error], midpoints))
        midpoints.append(choose_point(clusters[most_elements], midpoints))
    last_midpoints = midpoints
    clusters = create_clusters(points, midpoints)
    midpoints = [weighted_middle(cluster) for cluster in clusters]
    while not np.array_equal(last_midpoints, midpoints):
        last_midpoints = midpoints
        clusters = create_clusters(points, midpoints)
        midpoints = [weighted_middle(cluster) for cluster in clusters]
    return midpoints, [variance(clusters[i], midpoints[i]) for i in range(len(clusters))]


def choose_point(points, not_in):
    new = random.choice(points)
    while any(np.array_equal(new, p) for p in not_in):
        new = random.choice(points)
    return new


def variance(cluster, midpoint):
    if len(cluster) == 0:
        return 1
    else:
        return np.sum(np.linalg.norm(np.array(cluster) - midpoint, axis=1)**2) / len(cluster)


def max_comp(a, b):
    return a > b


def min_comp(a,b):
    return a < b


def find(list, func, comparer):
    max = 0
    max_value = func(list[0], 0)
    for i in range(1, len(list)):
        val = func(list[i], i)
        if comparer(val, max_value):
            max = i
            max_value = val
    return max_value, max


def create_clusters(points, midpoints):
    clusters = [[] for _ in range(len(midpoints))]
    for p in points:
        _, nearest = find(midpoints, lambda mp, i: np.linalg.norm(p - mp), min_comp)
        clusters[nearest].append(p)
    return clusters


def weighted_middle(points):
    return np.sum(points, axis=0) / len(points)
