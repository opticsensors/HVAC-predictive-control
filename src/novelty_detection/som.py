import numpy as np

class SOM():
    """
    SOM class for Self-Organizing Maps.
    """
    def __init__(self, 
                 som_grid_size=(10,10), 
                 learning_rate_0=1, 
                 sigma_0=1, 
                 max_iter=10000,
                 max_distance=4,
                 sigma_decay=0.1,
                 learning_rate_decay=0.1,
                 random_state=0,
                 methods={'init_som': 'uniform',
                          'bmu_distance': 'cityblock',
                          'compute_neighborhood': 'ceil',
                          'update_sigma_and_learning_rate': 'linear'}):
        
        """Initializes a Self Organizing Maps.

        Parameters:
        - som_grid_size (tuple): Size of the SOM grid.
        - learning_rate_0 (float): Initial learning rate.
        - sigma_0 (float): Initial sigma.
        - max_iter (int): Maximum number of iterations.
        - max_distance (int): Maximum distance neighbourhood.
        - sigma_decay (float): Sigma decay rate.
        - learning_rate_decay (float): Learning rate decay rate.
        - random_state (int): Random seed for reproducibility.
        - methods (dict): Dictionary specifying initialization and computation methods.
        """

        # Initialize descriptive features of SOM
        self.som_grid_size = som_grid_size
        self.learning_rate_0 = learning_rate_0 #initial value of learning rate
        self.sigma_0 = sigma_0 # initial value of sigma
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
        Find the Best Matching Unit (BMU) for the input vector X_row.

        Parameters:
        - X_row (numpy.ndarray): Input vector.
        - som (numpy.ndarray): Self-Organizing Map.

        Returns:
        - tuple: Coordinates of the BMU.
        """
        # min coordinates are the same for distance and the square of distance 
        distances = np.square(som - X_row).sum(axis=2)
        bmu_coordinates = np.unravel_index(distances.argmin().astype(int), self.som_grid_size)
        
        return bmu_coordinates
    
    def init_som(self, som_size, distribution='uniform'):
        """
        Initialize the weights of the SOM.

        Parameters:
        - som_size (tuple): Size of the SOM.
        - distribution (str): Type of distribution for weight initialization.

        Returns:
        - numpy.ndarray: Initialized SOM.
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
        Compute distance from BMU coordinates to the rest of the SOM grid cell coordinates.

        Parameters:
        - bmu (tuple): BMU coordinates.
        - distance (str): Type of distance metric.

        Returns:
        - numpy.ndarray: BMU distances.
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
        Compute the neighborhood map that multiplies the learning matrix.

        Parameters:
        - bmu_distance (numpy.ndarray): BMU distances.
        - sigma (float): Current value of sigma.
        - method (str): Method for computing the neighborhood map.

        Returns:
        - numpy.ndarray: Neighborhood map.
        """
        if method=='ceil':
            neighborhood_range = np.ceil(sigma * self.max_distance)
            neighborhood = np.zeros(shape=self.som_grid_size)
            neighborhood[bmu_distance <= neighborhood_range] = 1

        if method=='half':
            neighborhood = 1 / (2 ** bmu_distance)

        elif method=='exp':
            neighborhood = np.exp(-(bmu_distance ** 2/ (2*self.sigma_0 ** 2)))
        
        elif method=='mult':
            neighborhood = sigma * bmu_distance
        
        return neighborhood[..., np.newaxis]
    
    def update_sigma_and_learning_rate(self, sigma, learning_rate, n_iter, method='linear'):
        """
        Update sigma and learning rate based on the current iteration.

        Parameters:
        - n_iter (int): Current iteration.
        - method (str): Method for updating sigma and learning rate.

        Returns:
        - tuple: Updated sigma and learning rate.
        """
        if method=='exp':
            sigma = self.sigma_0 * np.exp(-(n_iter * self.sigma_decay))
            learning_rate = self.learning_rate_0 * np.exp(-(n_iter * self.learning_rate_decay))

        elif method=='linear':
            linear_decay =  1.0 - (n_iter/self.total_iterations)
            sigma = self.sigma_0 * linear_decay
            learning_rate = self.learning_rate_0  * linear_decay
        
        elif method=='constant':
            sigma = self.sigma_0
            learning_rate = self.learning_rate_0

        elif method=='mini_som':
            sigma = sigma / (1+self.sigma_decay/(self.max_iter/2))
            learning_rate = learning_rate / (1+self.learning_rate_decay/(self.max_iter/2))

        return sigma, learning_rate

    def update_som(self, X_row, som, learning_rate, sigma):
        """
        Update the SOM given an input vector.

        Parameters:
        - X_row (numpy.ndarray): Input vector.
        - som (numpy.ndarray): Self-Organizing Map.
        - learning_rate (float): Learning rate.
        - sigma (float): Sigma.

        Returns:
        - numpy.ndarray: Updated SOM.
        """
        # Find location (coordinates) of best matching unit
        bmu = self.find_bmu(X_row, som)

        bmu_distance = self.bmu_distance(bmu, 
                                         self.methods['bmu_distance'])

        # Compute update neighborhood
        neighborhood = self.compute_neighborhood(bmu_distance, 
                                                 sigma, 
                                                 self.methods['compute_neighborhood'])

        # Multiply by difference between input and weights (learning matrix)
        delta = learning_rate * neighborhood * (X_row - som)

        # Update weights
        som += delta
        return som

    def fit(self, X, epochs=None):
        """
        Train the SOM using input data for a specified number of epochs.
        If Epochs is None, it does a total of max_iter of random samples.

        Parameters:
        - X (numpy.ndarray): Input data (2D array).
        - epochs (int): Number of training epochs.

        Returns:
        - SOM: Trained SOM object.
        """
        # Define and init variables
        global_iter_counter = 0
        n_samples = X.shape[0]
        dim = X.shape[1]
        self.som_size = self.som_grid_size + (dim,)
        som = self.init_som(self.som_size, self.methods['init_som'])
        learning_rate = self.learning_rate_0
        sigma = self.sigma_0

        if epochs is None:
            self.total_iterations = self.max_iter

            for global_iter_counter in range(self.max_iter):
                # Train
                idx = self.rng.integers(n_samples)
                X_row = X[idx]
                # Do one step of training
                som = self.update_som(X_row, som, learning_rate, sigma)
                # Update learning rate and sigma
                sigma, learning_rate = self.update_sigma_and_learning_rate(sigma, learning_rate, global_iter_counter, 
                                                                    self.methods['update_sigma_and_learning_rate'])
        else:
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
                    som = self.update_som(X_row, som, learning_rate, sigma)
                    # Update learning rate and sigma
                    global_iter_counter += 1
                    sigma, learning_rate = self.update_sigma_and_learning_rate(sigma, learning_rate, global_iter_counter, 
                                                                            self.methods['update_sigma_and_learning_rate'])
        # Save som, learning rate and sigma after training
        self.som = som
        self.learning_rate_final = learning_rate
        self.sigma_final = sigma

        return self
    
