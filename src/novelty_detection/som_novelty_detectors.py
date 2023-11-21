import numpy as np
from sklearn.neighbors import NearestNeighbors
from novelty_detection.som_metrics import SOMmetrics

class KNN(SOMmetrics):
    """"
    This class uses provides an implementation of SOM for anomaly detection.
    """

    def __init__(
                self,
                som,
                min_number_per_bmu=1,
                number_of_neighbors=3,
                ):

        super().__init__(som)
        self.min_number_per_bmu = min_number_per_bmu
        self.number_of_neighbors = number_of_neighbors

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
    

class Quantization_Error(SOMmetrics):
    """"
    This class uses provides an implementation of SOM for anomaly detection.
    """

    def __init__(
                self,
                som,
                method='worst'
                ):

        super().__init__(som)
        self.method = method

    def find_dmax(self, df, freq_map, method):
        if method == 'worst':
            dmax = np.max(df['dmin'])
            dmax_map = np.full(self.som_grid_size, dmax)

        elif method == 'by_bmu':
            _ , dmax_map = self.find_dmin_min_and_max_by_units(df)

        #elif method == 'by_bmu_and_freq': #TODO

        return dmax_map

    def evaluate(self, X_train, X_test):
        X_test = np.atleast_2d(X_test)
        df_train = self.find_bmu_and_dmin(X_train, self.som)
        self.freq_map = self.find_bmu_counts(X_train, self.som)
        self.dmax_map = self.find_dmax(df_train, self.freq_map, self.method)
        Eq_training = np.sum(df_train['dmin']) / X_train.shape[0]
        df_test = self.find_bmu_and_dmin(X_test, self.som)
        Eq = df_test['dmin']
        Nd = np.zeros((X_test.shape[0],))
        dmax_training = self.dmax_map[df_test['row'], df_test['col']]
        Nd[Eq<=Eq_training] = 1
        condition = (Eq > Eq_training) & (Eq < dmax_training)
        Nd[condition] = 1 - (Eq[condition] - Eq_training) / (dmax_training[condition]  - Eq_training)
        Nd[Eq>dmax_training] = 0

        return Nd 
    