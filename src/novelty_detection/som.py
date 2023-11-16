import numpy as np
from sklearn.neighbors import NearestNeighbors

class SOM():
    def __init__(self, 
                 som_grid_size=(10,10), 
                 learning_rate=1, 
                 sigma=1, 
                 max_iter=10000,
                 max_distance=4,
                 sigma_decay=0.1,
                 learning_rate_decay=0.1,
                 random_state=0,
                 methods={'init_som': 'uniform',
                          'bmu_distance': 'cityblock',
                          'compute_neighborhood': 'ceil',
                          'update_sigma_and_learning_rate': 'linear'}):

        # Initialize descriptive features of SOM
        self.som_grid_size = som_grid_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.max_iter = max_iter
        self.max_distance = max_distance
        self.sigma_decay = sigma_decay
        self.learning_rate_decay = learning_rate_decay
        self.random_state = random_state
        self.methods = methods
        self.rng = np.random.default_rng(random_state) #np.random.seed(random_state)

        # coordinates of the som grid cells
        self.som_grid_coordinates = np.argwhere(np.ones(shape=self.som_grid_size))
        
    def find_bmu(self, X_row, som):
        """
        Find the index of the best matching unit for the input vector X_row.
        """
        # min coordinates are the same for distance and the square of distance 
        distances = np.square(som - X_row).sum(axis=2)
        bmu_coordinates = np.unravel_index(distances.argmin().astype(int), self.som_grid_size)
        
        return bmu_coordinates
    
    def init_som(self, som_size, distribution='uniform'):
        """
        Initialize the weights of som (3d array)
        """
        if distribution=='normal':
            som = self.rng.normal(size=som_size)

        elif distribution=='sq_normal':
            som = self.rng.normal(size=som_size)**2 -1

        elif distribution=='uniform':
            som = self.rng.uniform(size=som_size)

        return som
    
    def bmu_distance(self, bmu, distance='cityblock'):
        """
        Computes distance from bmu coordinates to the rest of the som grid cell coordinates
        """
        bmu_2darray = np.atleast_2d(bmu)

        if distance=='cityblock':
            bmu_distance = np.abs(bmu_2darray[:,None] - self.som_grid_coordinates).sum(axis=2)

        elif distance=='euclidian':
            bmu_distance = np.linalg.norm(bmu_2darray - self.som_grid_coordinates, axis=1)

        bmu_distance = bmu_distance.reshape(self.som_grid_size[0], self.som_grid_size[1])

        return bmu_distance
    
    def compute_neighborhood(self, bmu_distance, sigma, method='ceil'):
        """
        Computes the neighborhood map that multiplies the learning matrix
        """
        if method=='ceil':
            neighborhood_range = np.ceil(sigma * self.max_distance)
            neighborhood = np.zeros(shape=self.som_grid_size)
            neighborhood[bmu_distance <= neighborhood_range] = 1

        elif method=='exp':
            neighborhood = np.exp(-(bmu_distance ** 2/ (2*self.sigma ** 2)))
        
        elif method=='mult':
            neighborhood = sigma * bmu_distance
        
        return neighborhood[..., np.newaxis]
    
    def update_sigma_and_learning_rate(self, n_iter, method='linear'):

        if method=='exp':
            sigma = self.sigma * np.exp(-(n_iter * self.sigma_decay))
            learning_rate = self.learning_rate * np.exp(-(n_iter * self.learning_rate_decay))

        elif method=='linear':
            sigma = 1.0 - (n_iter/self.total_iterations)
            learning_rate = sigma * self.learning_rate

        return sigma, learning_rate

    def step(self, X_row, learning_rate, sigma):
        """
        Do one step of training on the given input vector.
        """
        # Find location (coordinates) of best matching unit
        bmu = self.find_bmu(X_row, self.som)

        bmu_distance = self.bmu_distance(bmu, 
                                         self.methods['bmu_distance'])

        # Compute update neighborhood
        neighborhood = self.compute_neighborhood(bmu_distance, 
                                                 sigma, 
                                                 self.methods['compute_neighborhood'])

        # Multiply by difference between input and weights (learning matrix)
        delta = learning_rate * neighborhood * (X_row - self.som)

        # Update weights
        self.som += delta

    def fit_with_epochs(self, X, epochs=1):
        """
        Take data (2d array) as input and fit the SOM to that
        data for the specified number of epochs.
        """
        # Count total number of iterations
        global_iter_counter = 0
        n_samples = X.shape[0]
        dim = X.shape[1]
        self.som_size = self.som_grid_size + (dim,)
        self.som = self.init_som(self.som_size, self.methods['init_som'])
        learning_rate = self.learning_rate
        sigma = self.sigma
        self.total_iterations = np.minimum(epochs * n_samples, self.max_iter)

        for epoch in range(epochs):
            # Break if past max number of iterations
            if global_iter_counter > self.max_iter:
                break

            indices = self.rng.permutation(n_samples)

            # Train
            for idx in indices:
                # Break if past max number of iterations
                if global_iter_counter > self.max_iter:
                    break
                X_row = X[idx]
                # Do one step of training
                self.step(X_row, learning_rate, sigma)
                # Update learning rate and sigma
                global_iter_counter += 1
                sigma, learning_rate = self.update_sigma_and_learning_rate(global_iter_counter, 
                                                                           self.methods['update_sigma_and_learning_rate'])

        return self

    def fit_with_iterations(self, X):
        """
        Take data (2d array) as input and fit the SOM to that
        data for the specified number of epochs.
        """
        # Count total number of iterations
        global_iter_counter = 0
        n_samples = X.shape[0]
        dim = X.shape[1]
        self.som_size = self.som_grid_size + (dim,)
        self.som = self.init_som(self.som_size, self.methods['init_som'])
        learning_rate = self.learning_rate
        sigma = self.sigma
        self.total_iterations = self.max_iter

        for global_iter_counter in range(self.max_iter):
            # Train
            idx = self.rng.integers(n_samples)
            X_row = X[idx]
            # Do one step of training
            self.step(X_row, learning_rate, sigma)
            # Update learning rate and sigma
            global_iter_counter += 1
            sigma, learning_rate = self.update_sigma_and_learning_rate(global_iter_counter, 
                                                                       self.methods['update_sigma_and_learning_rate'])

        return self
    
class KNN_NoveltyDetection():
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

    def find_bmu(self, X_row, som):
        """
        Find the index of the best matching unit for the input vector X_row.
        """
        # min coordinates are the same for distance and the square of distance 
        distances = np.square(som - X_row).sum(axis=2)
        bmu_coordinates = np.unravel_index(distances.argmin().astype(int), self.som_grid_size)
        
        return bmu_coordinates

    def find_bmu_counts(self, X):
        """
        This functions maps a training set to the fitted network and evaluates for each
        node in the SOM the number of evaluations mapped to that node. This gives counts per BMU.
        :param training_data: numpy array of training data
        :return: An array of the same shape as the network with the best matching units.
        """
        som_grid_shape = (self.som.shape[0], self.som.shape[1])
        bmu_counts = np.zeros(shape=som_grid_shape)
        for X_row in X:
            bmu = self.find_bmu(X_row, self.som)
            bmu_counts[bmu] += 1
        return bmu_counts

    def fit(self, X):
        """
        This function fits the anomaly detection model to some training data.
        It removes nodes that are too sparse by the minNumberPerBmu threshold.
        """
        bmu_counts = self.find_bmu_counts(X)
        self.bmu_counts = bmu_counts.flatten()
        self.allowed_nodes = self.som[bmu_counts >= self.min_number_per_bmu]
        return self

    def evaluate(self, X):
        """
        This function maps the evaluation data to the previously fitted network. It calculates the
        anomaly measure based on the distance between the observation and the K-NN nodes of this
        observation.
        """
        try:
            assert self.allowed_nodes.shape[0] > 1
        except AssertionError:
            raise Exception(
                "There are no nodes satisfying the minimum criterium, algorithm cannot proceed."
            )
        else:
            classifier = NearestNeighbors(n_neighbors=self.number_of_neighbors)
            classifier.fit(self.allowed_nodes)
            dist, _ = classifier.kneighbors(X)
        return dist.mean(axis=1)