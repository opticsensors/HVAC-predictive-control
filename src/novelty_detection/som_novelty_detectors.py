import numpy as np
from sklearn.neighbors import NearestNeighbors
from novelty_detection.utils import argmin_first_two_axes

class KNN():
    """"
    This class uses provides an implementation of SOM for anomaly detection.
    """

    def __init__(
                self,
                som,
                min_number_per_bmu=1,
                number_of_neighbors=3,
                ):

        self.som = som
        self.min_number_per_bmu = min_number_per_bmu
        self.number_of_neighbors = number_of_neighbors
        self.som_grid_size = (som.shape[0], som.shape[1])
        self.som_size = som.shape

    def find_bmu(self, X_row, som):
        """
        Find the index of the best matching unit for the input vector X_row.
        """
        # min coordinates are the same for distance and the square of distance 
        distances = np.square(som - X_row).sum(axis=2)
        bmu_coordinates = np.unravel_index(distances.argmin().astype(int), self.som_grid_size)
        
        return bmu_coordinates
    
    def find_bmu_broadcasted(self, X, som):
        #generate a distance map for every X row (every third dim belongs to a row)
        som_bro = som[:,:,:,np.newaxis]
        data_bro = X.T[np.newaxis,np.newaxis, :, :]
        distance_map = np.linalg.norm(som_bro-data_bro, axis=2)
        # find the min distance indices for every third dim 
        return argmin_first_two_axes(distance_map)

    def find_bmu_counts(self, X, som):
        """
        This functions maps a training set to the fitted network and evaluates for each
        node in the SOM the number of evaluations mapped to that node. This gives counts per BMU.
        """
        bmu_counts = np.zeros(shape=self.som_grid_size)
        for X_row in X:
            bmu = self.find_bmu(X_row, som)
            bmu_counts[bmu] += 1
        return bmu_counts
    
    def find_bmu_counts_broadcasted(self, X, som):
        bmu_bro = self.find_bmu_broadcasted(X, som)
        unique_bmu, counts_idx = np.unique(bmu_bro, return_counts=True, axis=0)
        bmu_counts = np.zeros(shape=(10,10))
        bmu_counts[unique_bmu[:,0], unique_bmu[:,1]] = counts_idx
        return bmu_counts

    def evaluate(self, X):
        """
        This function maps the evaluation data to the previously fitted network. It calculates the
        anomaly measure based on the distance between the observation and the K-NN nodes of this
        observation.
        """
        bmu_counts = self.find_bmu_counts(X, self.som)
        allowed_nodes = self.som[bmu_counts >= self.min_number_per_bmu]

        try:
            assert allowed_nodes.shape[0] > 1
        except AssertionError:
            raise Exception(
                "There are no nodes satisfying the minimum criterium, algorithm cannot proceed."
            )
        else:
            classifier = NearestNeighbors(n_neighbors=self.number_of_neighbors)
            classifier.fit(allowed_nodes)
            dist, _ = classifier.kneighbors(X)

        return dist.mean(axis=1)
    

class Quantization_Error():
    """"
    This class uses provides an implementation of SOM for anomaly detection.
    """

    def __init__(
                self,
                som,
                ):

        self.som = som
        self.som_grid_size = (som.shape[0], som.shape[1])
        self.som_size = som.shape

    def quantization_error(self, X, som):
        X = np.atleast_2d(X)
        total_error = 0.0
        for row in X:
            distance_map = np.linalg.norm(som-row, axis=2)
            total_error += distance_map.min()
        quantization_err = total_error / X.shape[0]
        return quantization_err

    def quantization_error_broadcasted(self, X, som):
        X = np.atleast_2d(X)
        som_bro = som[:,:,:,np.newaxis]
        data_bro = X.T[np.newaxis,np.newaxis, :, :]
        distance_map = np.linalg.norm(som_bro-data_bro, axis=2)
        total_error = np.min(distance_map, axis=(0,1)).sum()
        quantization_err = total_error / X.shape[0]
        return quantization_err
    
    def find_dmin(self, X, som):
        X = np.atleast_2d(X)
        l_dmin = []

        for row in X:
            distance_map = np.linalg.norm(som-row, axis=2)
            l_dmin.append(distance_map.min())
    
        return l_dmin

    def evaluate(self, X_train, X_test):
        l_dmin = self.find_dmin(X_train, self.som)
        dmax_training = np.max(l_dmin) - np.min(l_dmin)
        Eq_training = np.sum(l_dmin) / X_train.shape[0]
        Eq = self.find_dmin(X_test, self.som)
        Eq = np.array(Eq)
        Nd = np.zeros((X_test.shape[0],))
        Nd[Eq<=Eq_training] = 1
        condition = (Eq > Eq_training) & (Eq < dmax_training)
        Nd[condition] = 1 - (Eq[condition] - Eq_training) / (dmax_training - Eq_training)
        Nd[Eq>dmax_training] = 0
        return Nd
    