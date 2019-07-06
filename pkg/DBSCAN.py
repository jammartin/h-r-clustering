import numpy as np


def d(x, y):
    return np.sqrt(np.sum((x - y)**2))


class DBSCAN:
    def __init__(self, data, norm=d):
        # private fields
        self.__data = data
        self.__eps = 0.1
        self.__minPts = 10
        self.__visited = None
        self.__norm = norm
        # public fields
        self.clusters = None

    # private methods
    def __is_core_point(self, x):
        # get all density reachable points
        neighbors = [i for i, y in enumerate(self.__data) if self.__norm(x, y) < self.__eps]
        if len(neighbors) >= self.__minPts:
            return True, neighbors
        else:
            return False, neighbors

    def __get_neighbors_of_core_points(self, points_i):
        neighbors_of_core_points = []  # return value
        for i in points_i:
            is_core_point, neighbors = self.__is_core_point(self.__data[i])
            if is_core_point:
                # append new neighbors
                neighbors_of_core_points += [nghbr for nghbr in neighbors if nghbr not in neighbors_of_core_points]
        return neighbors_of_core_points

    def __get_density_reachable_points(self, core_point, neighbors):
        density_reachable_points = [core_point]  # return value
        while len(neighbors):
            density_reachable_points += neighbors
            neighbors_of_core_points = self.__get_neighbors_of_core_points(neighbors)
            neighbors = [i for i in neighbors_of_core_points if i not in density_reachable_points]
        return np.array(density_reachable_points)

    def __visit_point(self, i, x):
        self.__visited[i] = 1
        is_core_point, neighbors = self.__is_core_point(x)
        if is_core_point:
            print("A core point has been found. Creating cluster.")
            self.clusters[0].remove(i)
            density_reachable_points = self.__get_density_reachable_points(i, neighbors)
            self.__visited[density_reachable_points] = 1  # mark all points in cluster as visited
            self.clusters.append(density_reachable_points)  # store cluster
            print(len(self.clusters[-1]), "points have been assigned to cluster", len(self.clusters) - 1)

    # public methods
    def run(self, eps, minPts):
        self.__eps = eps
        self.__minPts = minPts
        self.__visited = np.zeros(len(self.__data), dtype=np.int8)
        self.clusters = [[i for i in range(len(self.__data))]]  # first "cluster" holds unassigned points

        # main loop
        print("Clustering with eps =", self.__eps, "and minPts =", self.__minPts, "...")
        for i, x in enumerate(self.__data):
            if not self.__visited[i]:
                self.__visit_point(i, x)
        # convert python built in list to numpy array
        self.clusters[0] = np.array(self.clusters[0])
        self.clusters = np.array(self.clusters)
        print("...done.")

    def plot_2d(self, ax, markersize=3):
        # plot unassigned points
        unassigned_points = np.take(self.__data, self.clusters[0], axis=0)
        ax.plot(unassigned_points[:, 0], unassigned_points[:, 1], 'k.', markersize=markersize)
        for cluster in self.clusters[1:]:
            cluster_points = np.take(self.__data, cluster, axis=0)
            ax.plot(cluster_points[:, 0], cluster_points[:, 1], '.', markersize=markersize)
