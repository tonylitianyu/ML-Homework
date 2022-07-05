import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

        self.feature_labels = []

    def get_distance(self, p1, p2):
        arr_diff = p1 - p2

        return np.sqrt(np.sum(np.square(arr_diff)))



    def updates(self, features):
        mean_arr = []

        for c in range(self.n_clusters):
            mean_arr.append([])

        for i in range(len(features)):
            min_m_idx = 0
            min_m_val = self.get_distance(features[i], self.means[0])

            for m in range(len(self.means)):
                dis = self.get_distance(features[i], self.means[m])
                if dis < min_m_val:
                    min_m_val = dis
                    min_m_idx = m

            mean_arr[min_m_idx].append(features[i])

        for m in range(len(self.means)):
            curr_m_arr = np.array(mean_arr[m])
            if len(curr_m_arr) > 0:
                curr_mean = np.mean(curr_m_arr, axis=0)
                self.means[m] = curr_mean




    def get_mean_epsilon(self, last_mean):
        dis_sum = 0.0
        for m in range(len(self.means)):
            dis = self.get_distance(last_mean[m], self.means[m])
            dis_sum += dis

            


        return dis_sum



    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """

        self.feature_labels = np.zeros(len(features))

        first_mean_idx = np.random.choice(features.shape[0], self.n_clusters, replace=False)

        self.means = features[first_mean_idx]


        last_mean = self.means.copy()
        eps = 100.0

        while eps > 0.001:

            self.updates(features)

            eps = self.get_mean_epsilon(last_mean)


            last_mean = self.means.copy()


    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        
        predictions = np.zeros(len(features))

        for i in range(len(features)):
            min_m_idx = 0
            min_m_val = self.get_distance(features[i], self.means[0])
            for m in range(0,len(self.means)):
                dis = self.get_distance(features[i], self.means[m])

                if dis < min_m_val:
                    min_m_val = dis
                    min_m_idx = m


            predictions[i] = min_m_idx

        return predictions

