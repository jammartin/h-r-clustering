import warnings as wrn
import numpy as np


class K_means:
    def __init__(self, data):
        # private fields
        self.__data = data
        self.__assignments = None
        self.__means = None
        # public fields
        self.assignments = None
        self.means = None
        np.random.seed()

    # private methods
    def __objective_function(self):
        J = 0.
        for k, mean in enumerate(self.__means):
            J += np.sum((np.take(self.__data, np.nonzero(self.__assignments[:, k])[0], axis=0) - mean) ** 2)
        return J

    def __expectation_step(self):
        success = True
        for k in range(len(self.__means)):
            N_k = np.count_nonzero(self.__assignments[:, k])

            if N_k == 0:
                wrn.warn("A cluster has become empty. The initial values of the cluster means are choosen poorely."
                         , RuntimeWarning)
                success = False
                break
            else:
                self.__means[k] = 1. / N_k * np.sum(
                    np.take(self.__data, np.nonzero(self.__assignments[:, k])[0], axis=0), axis=0)
        return success

    def __maximization_step(self):
        updated_assignments = np.zeros(self.__assignments.shape, dtype=np.int8)
        for i, x in enumerate(self.__data):
            # calculate squared distances of a point to each of the cluster centers
            d_x_mu = np.array([np.sum((x - mu) ** 2) for mu in self.__means])
            k_min = np.argmin(d_x_mu)
            if type(k_min) is np.ndarray and len(k_min) > 1:
                wrn.warning("A data point is closest to more than one cluster mean. Taking the first one."
                            , RuntimeWarning)
                k_min = k_min[0]
            updated_assignments[i, k_min] = 1
        if np.array_equal(self.__assignments, updated_assignments):
            return True  # converged
        else:
            self.__assignments = updated_assignments
            return False  # not converged

    # public methods
    def run(self, n_clusters, n_init=10):

        # initialize objective function
        J_min = np.inf

        for i_init in range(n_init):

            aborted = True
            while aborted:
                print("Random initialization of cluster means", i_init + 1, "/", n_init)
                # container for the cluster assignments
                self.__assignments = np.zeros((len(self.__data), n_clusters), dtype=np.int8)

                # initialize with random means
                self.__means = np.zeros((n_clusters, self.__data.shape[1]))
                # generate random position in each feature space
                for i_ft in range(self.__data.shape[1]):
                    self.__means[:, i_ft] = np.random.uniform(low=np.amin(self.__data[:, i_ft])
                                                              , high=np.amax(self.__data[:, i_ft])
                                                              , size=n_clusters)
                it = 0
                # main loop
                while not self.__maximization_step():
                    if not self.__expectation_step():
                        print("Aborting for this set of initial values for the cluster means.")
                        aborted = True
                        break
                    else:
                        aborted = False
                    it += 1

            J = self.__objective_function()
            print("Objective function J =", J)
            print("Total number of iterations:", it)

            # store assignments and cluster means with lowest J
            if J < J_min:
                self.assignments = np.copy(self.__assignments)
                self.means = np.copy(self.__means)
                J_min = J

    def plot_2d(self, ax):
        # plot data with clusters in different colors
        for k in range(len(self.means)):
            cluster_data = np.take(self.__data, np.nonzero(self.assignments[:, k])[0], axis=0)
            ax.plot(cluster_data[:, 0], cluster_data[:, 1], '.', markersize=1)
        # plot cluster means
        ax.plot(self.means[:, 0], self.means[:, 1], 'kx', markersize=8, mew=3)
