import numpy as np
import argparse
import scipy.io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class MykmeansClustering:
    def __init__(self, dataset_file):
        self.model = None
        self.data = None

        self.dataset_file = dataset_file
        self.read_mat()

    def read_mat(self):
        mat = scipy.io.loadmat(self.dataset_file)
        self.data = mat['X']
        
    def model_fit(self):
        '''
        initialize self.model here and execute kmeans clustering here
        '''
        num_clusters = 4
        max_iter = 500

        self.model = KMeans(n_clusters = num_clusters, max_iter = max_iter)
        self.model.fit(self.data)
        
        cluster_centers = np.array([[0,0]])
        cluster_centers = self.model.cluster_centers_
        return cluster_centers
    
    def plotData(self):
        labels = self.model.labels_
        
        # Get the cluster centers
        cluster_centers = self.model.cluster_centers_
        
        # Plot the data points, colored by their cluster label
        plt.figure(figsize=(8, 6))
        plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='viridis', s=50)
        
        # Plot the cluster centers with a different marker and color
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='X', label='Centers')
        
        plt.title('K-means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kmeans clustering')
    parser.add_argument('-d','--dataset_file', type=str, default = "dataset_q2.mat", help='path to dataset file')
    args = parser.parse_args()
    classifier = MykmeansClustering(args.dataset_file)
    clusters_centers = classifier.model_fit()
    print(clusters_centers)
    classifier.plotData()