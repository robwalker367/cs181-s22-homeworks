# CS 181, Spring 2022
# Homework 4

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Loading datasets for K-Means and HAC
small_dataset = np.load("P2_Autograder_Data.npy") # np.load("data/small_dataset.npy")
large_dataset = np.load("data/large_dataset.npy")

np.random.seed(2)

# NOTE: You may need to add more helper functions to these classes
class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K, runs):
        self.K = K
        self.means = None
        self.runs = runs
        self.assignments = None
        self.losses = np.zeros(runs)

    # X is a (N x 784) array since the dimension of each image is 28x28.
    def fit(self, X):
        # Initialize cluster assignments and random centers
        self.assignments = np.zeros(X.shape[0])
        self.means = np.random.randn(self.K, X.shape[1])

        for run in range(self.runs):
            # Assign datapoints to nearest cluster
            for i, x in enumerate(X):
                self.assignments[i] = np.argmin(np.linalg.norm(x - self.means, axis=1) ** 2)

            # Update cluster means
            for k in range(self.K):
                clusteroid = X[self.assignments == k]
                if clusteroid.size != 0:
                    self.means[k] = clusteroid.mean(axis=0)
            
            # Record loss
            self.losses[run] = self.__objective(X)
        return

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.means
    
    def get_losses(self):
        return self.losses
    
    def __objective(self, X):
        return np.sum([np.linalg.norm(x - self.means[int(self.assignments[i])]) ** 2 for i, x in enumerate(X)])

class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage
        self.assignments = []
        self.X = None
    
    # X is a (N x 784) array since the dimension of each image is 28x28.
    def fit(self, X):
        self.X = X

        # Assign each x in X to its own cluster
        N = X.shape[0]

        assignment = np.arange(N)
        self.assignments.append(np.copy(assignment))

        nclusters = np.unique(assignment).size

        while nclusters > 1:
            if self.linkage == 'centroid':
                # Find nearest clusters
                midx, mval = [0, 1], float('inf')
                for i in range(N):
                    for j in range(i+1, N):
                        Xi, Xj = X[assignment == i], X[assignment == j]
                        if Xi.size == 0 or Xj.size == 0:
                            continue
                        m = np.linalg.norm(Xi.mean(axis=0) - Xj.mean(axis=0))
                        if m < mval:
                            mval = m
                            midx = [i, j]

                # Merge clusters
                assignment[assignment == midx[1]] = midx[0]
                self.assignments.append(np.copy(assignment))

                print(midx)
                print(nclusters)
                nclusters -= 1
            else:
                return

        return

    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self, n_clusters):
        assignment = self.assignments[-n_clusters]
        means = np.zeros((n_clusters, self.X.shape[1]))
        clusters = np.unique(assignment)
        for i, cluster in enumerate(clusters):
            means[i] = self.X[assignment == cluster].mean(axis=0)
        return means

# Plotting code for parts 2 and 3
def make_mean_image_plot(data, standardized=False):
    # Number of random restarts
    niters = 3
    runs = 10
    K = 10
    # Will eventually store the pixel representation of all the mean images across restarts
    allmeans = np.zeros((K, niters, 784))
    alllosses = np.zeros((K, runs))
    for i in range(niters):
        KMeansClassifier = KMeans(K=K, runs=runs)
        KMeansClassifier.fit(data)
        allmeans[:,i] = KMeansClassifier.get_mean_images()
        alllosses[:,i] = KMeansClassifier.get_losses()
    
    # Plot losses
    fig = plt.figure()
    for i in range(niters):
        plt.plot(alllosses[:,i], label=f'Run {i}')
    plt.title('Losses on each run')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10,10))
    plt.suptitle('Class mean images across random restarts' + (' (standardized data)' if standardized else ''), fontsize=16)
    for k in range(K):
        for i in range(niters):
            ax = fig.add_subplot(K, niters, 1+niters*k+i)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            if k == 0: plt.title('Iter '+str(i))
            if i == 0: ax.set_ylabel('Class '+str(k), rotation=90)
            plt.imshow(allmeans[k,i].reshape(28,28), cmap='Greys_r')
    plt.show()

# ~~ Part 2 ~~
# make_mean_image_plot(large_dataset, False)

# ~~ Part 3 ~~
# Change this line! standardize large_dataset and store the result in large_dataset_standardized
# std = np.std(large_dataset, axis=0)
# std[std == 0] = 1
# large_dataset_standardized = (large_dataset - large_dataset.mean(axis=0)) / std
# make_mean_image_plot(large_dataset_standardized, True)

# Plotting code for part 4
LINKAGES = ['centroid'] # [ 'max', 'min', 'centroid' ]
n_clusters = 10

fig = plt.figure(figsize=(10,10))
plt.suptitle("HAC mean images with max, min, and centroid linkages")
for l_idx, l in enumerate(LINKAGES):
    # Fit HAC
    hac = HAC(l)
    hac.fit(small_dataset)
    mean_images = hac.get_mean_images(n_clusters)
    # Make plot
    for m_idx in range(mean_images.shape[0]):
        m = mean_images[m_idx]
        ax = fig.add_subplot(n_clusters, len(LINKAGES), l_idx + m_idx*len(LINKAGES) + 1)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        if m_idx == 0: plt.title(l)
        if l_idx == 0: ax.set_ylabel('Class '+str(m_idx), rotation=90)
        plt.imshow(m.reshape(28,28), cmap='Greys_r')
plt.show()

# TODO: Write plotting code for part 5

# TODO: Write plotting code for part 6