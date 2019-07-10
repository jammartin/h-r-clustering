import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import k_means as sklearn_k_means
from sklearn.cluster import DBSCAN as sklearn_DBSCAN

# import local modules
from DBSCAN import DBSCAN
from K_means import K_means


# Set the default color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.rcsetup.cycler(color=['r', 'b', 'm', 'c', 'g'])
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


if __name__=="__main__":

    url = "https://raw.githubusercontent.com/astronexus/HYG-Database/master/hygdata_v3.csv"
    hyg_data = pd.read_csv(url, usecols=['ci', 'absmag']).query('ci < 2.5')
    print("Number of data points: ", len(hyg_data.index))

    plt.figure(figsize=(9, 12), dpi=200)
    plt.plot(hyg_data['ci'], hyg_data['absmag'], 'k.', markersize=1)
    ymin, ymax = plt.gca().get_ylim()  # getting limits for y-axis
    plt.ylim(ymax, ymin)  # inverting y-axis
    plt.title("Hertzsprung-Russel-Diagram")
    plt.xlabel(r"Color Index $E_{(B-V)}$")
    plt.ylabel(r"Absolute Magnitude $M$")
    plt.tight_layout()
    plt.savefig('../plots/h-r-diagram_unclustered.png')

    # scale and switch collumns to conform to the convention that the x axis is the 0th collumn
    data = np.zeros(hyg_data.shape)
    data.T[[0, 1]] = StandardScaler().fit_transform(hyg_data).T[[1, 0]]

    # K-MEANS
    n_clusters_lst = [2, 3, 5]

    # initialize our algorithm class with our given data
    k_means = K_means(data)

    fig, axs = plt.subplots(nrows=1, ncols=len(n_clusters_lst), figsize=(16, 9), dpi=200)
    for i_n_clusters, n_clusters in enumerate(n_clusters_lst):
        print("Clustering with", n_clusters, "distinct clusters ...")
        means, assignments, J = sklearn_k_means(data, n_clusters=n_clusters, init='random'
                                                , n_init=50, algorithm='full', tol=0.)
        print("...done. Final value of the objective function:", J)
        # feed the results to our K_means class in order to use their plot function
        k_means.means = means
        # convert the returned assignment into our format
        k_means.assignments = np.zeros((len(data), n_clusters), dtype=np.int8)
        for i, assignment in enumerate(assignments):
            k_means.assignments[i, assignment] = 1
        k_means.plot_2d(axs[i_n_clusters])  # plotting
        ymin, ymax = axs[i_n_clusters].get_ylim()  # getting limits for y-axis
        axs[i_n_clusters].set_ylim([ymax, ymin])  # inverting y-axis
        axs[i_n_clusters].title.set_text(r"Number of clusters $K = " + str(n_clusters) + r"$")
    plt.tight_layout()
    plt.savefig('../plots/h-r-diagrams_kmeans.png')

    # DBSCAN
    dbscan = DBSCAN(data)  # initialize our class with all the data in order to use it for plotting
    params_lst_full_data = [[0.3, 200], [0.1, 50], [0.2, 20]]  # array of [eps, minPts]

    fig, axs = plt.subplots(nrows=1, ncols=len(params_lst_full_data), figsize=(16, 9), dpi=200)
    for i_params, params in enumerate(params_lst_full_data):
        dbscan_model = sklearn_DBSCAN(eps=params[0], min_samples=params[1], metric='euclidean')  # initialize the model
        print("Clustering with eps =", params[0], "and minPts =", params[1], "...")
        dbscan_model.fit(data)  # run the algorithm
        print("Total number of clusters found", len(np.unique(dbscan_model.labels_)) - 1)
        print("...done.")
        # convert result to our format
        dbscan.clusters = [[] for _ in range(len(np.unique(dbscan_model.labels_)))]
        for i, cluster in enumerate(dbscan_model.labels_):
            if cluster == -1:
                cluster = 0
            else:
                cluster = cluster + 1
            dbscan.clusters[cluster].append(i)
        dbscan.plot_2d(axs[i_params], markersize=1)  # plotting
        ymin, ymax = axs[i_params].get_ylim()  # getting limits for y-axis
        axs[i_params].set_ylim([ymax, ymin])  # inverting y-axis
        axs[i_params].title.set_text(r"$\varepsilon = " + str(params[0])
                                     + r", \ \text{minPts} = " + str(params[1]) + r"$")
    plt.tight_layout()
    plt.savefig('../plots/h-r-diagrams_dbscan.png')

