import math

import numpy as np
from scipy.stats import multivariate_normal

#https://github.com/FlorisHoogenboom/som-anomaly-detector/tree/master/som_anomaly_detector

class KohonenSom(object):
    """
    This class provides an implementation of the Kohonen SOM algorithm.
    It supports SOMs of an arbitrary dimension, which may be handy for data quality purposes.
    """

    def __init__(
        self,
        shape,
        input_size,
        learning_rate,
        learning_decay=1,
        initial_radius=1,
        radius_decay=1,
    ):
        """ Initialization of the SOM

        :param dimension: The shape of the network. Each entrty in the tuple corresponds
            to one direction
        :type dimension: tuple of ints
        :param learning_rate: The inital learning rate.
        :type learning_rate: float, should be > 0
        :param initial_radius:  The initial radius.
        :type initial_radius: float, should be > 0
        """

        self.shape = shape
        self.dimension = shape.__len__()
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.initial_radius = initial_radius
        self.radius_decay = radius_decay

        # Initialize a distance matrix to avoid computing the distance on each iteration
        distance = np.fromfunction(
            self._distance_function, tuple(2 * i + 1 for i in shape)
        )

        gaussian_transorm = np.vectorize(
            lambda x: multivariate_normal.pdf(x, mean=0, cov=1)
        )
        self.distance = gaussian_transorm(distance)

        # We add an extra dimension so that we can easily perform multiplication later on
        self.distance = np.repeat(self.distance, self.input_size, self.dimension - 1)
        self.distance = np.reshape(
            self.distance, newshape=(distance.shape + (self.input_size,))
        )

        # Initialize the grid
        self.grid = np.random.rand(*(self.shape + (self.input_size,))) * 2 - 1

        return

    def reset(self):
        """
        This function resets the grid for a new estimation to take place
        :return: Nothing
        """
        self.grid = np.random.rand(*(self.shape + (self.input_size,))) * 2 - 1

        return

    def _distance_function(self, *args):
        """ Computes the euclidean distance for an arbitrary number of points
        :param points: arbitrary number of points
        :type points: float
        :return: the euclidean distance
        """
        # Fill the array in such a way it contains zero
        # distance at the center, i.e. index = shape
        return sum(
            [(i - x) ** 2 for i, x in zip(args, self.shape)]
        )


    def get_bmu(self, sample):
        """Find the best matchin unit for a specific sample

        :param sample: The data point for which the best matching unit should be found.
        :type sample: numpy.array
        :return: numpy.ndarray with index
        """

        distances = np.square(self.grid - sample).sum(axis=self.dimension)
        bmu_index = np.unravel_index(distances.argmin().astype(int), self.shape)
        return bmu_index

    def fit(self, training_sample, num_iterations):
        """Train the SOM to a specific dataset.

        :param training_sample: The complete training dataset
        :type training_sample: 2d ndarray
        :param num_iterations: The number of iterations used for training
        :type num_iterations: int
        :return: a reference to the object
        """

        sigma = self.initial_radius
        learning_rate = self.learning_rate
        for i in range(1, num_iterations):

            obs = training_sample[np.random.choice(training_sample.shape[0], 1)][0]
            bmu = self.get_bmu(obs)
            self.update_weights(obs, bmu, sigma, learning_rate)

            # Update the parameters to let them decay to 0
            sigma = self.initial_radius * math.exp(-(i * self.radius_decay))
            learning_rate = self.learning_rate * math.exp(-(i * self.learning_decay))
        return self

    def update_weights(self, training_vector, bmu, sigma, learning_speed):
        reshaped_array = self.grid.reshape((np.product(self.shape), self.input_size))

        # We want to roll the distance matrix such that we have the BMU at the center
        bmu_distance = self.distance
        for i, bmu_ind in enumerate(bmu):
            bmu_distance = np.roll(bmu_distance, bmu_ind, axis=i)

        # Next we take part of the second quadrant of the matrix since this corresponds to
        # the distance matrix we desire
        for i, shape_ind in enumerate(self.shape):
            slc = [slice(None)] * len(bmu_distance.shape)
            slc[i] = slice(shape_ind, 2 * shape_ind)
            bmu_distance = bmu_distance[slc]

        # Multiply by sigma to emulate a decreasing radius effect
        bmu_distance = sigma * bmu_distance

        learning_matrix = -(self.grid - training_vector)
        scaled_learning_matrix = learning_speed * (bmu_distance * learning_matrix)
        self.grid = self.grid + scaled_learning_matrix

        return
    
import numpy as np
from sklearn.neighbors import NearestNeighbors


class AnomalyDetection(KohonenSom):
    """"
    This class uses provides an specific implementation of Kohonnen Som for anomaly detection.
    """

    def __init__(
        self,
        shape,
        input_size,
        learning_rate,
        learning_decay=0.1,
        initial_radius=1,
        radius_decay=0.1,
        min_number_per_bmu=1,
        number_of_neighbors=3,
    ):
        super(AnomalyDetection, self).__init__(
            shape,
            input_size,
            learning_rate,
            learning_decay,
            initial_radius,
            radius_decay,
        )

        self.minNumberPerBmu = min_number_per_bmu
        self.numberOfNeighbors = number_of_neighbors
        return

    def get_bmu_counts(self, training_data):
        """
        This functions maps a training set to the fitted network and evaluates for each
        node in the SOM the number of evaluations mapped to that node. This gives counts per BMU.
        :param training_data: numpy array of training data
        :return: An array of the same shape as the network with the best matching units.
        """
        bmu_counts = np.zeros(shape=self.shape)
        for observation in training_data:
            bmu = self.get_bmu(observation)
            bmu_counts[bmu] += 1
        return bmu_counts

    def fit(self, training_data, num_iterations):
        """
        This function fits the anomaly detection model to some training data.
        It removes nodes that are too sparse by the minNumberPerBmu threshold.
        :param training_data: numpy array of training data
        :param num_iterations: number of iterations allowed for training
        :return: A vector of allowed nodes
        """
        self.reset()
        super(AnomalyDetection, self).fit(training_data, num_iterations)
        bmu_counts = self.get_bmu_counts(training_data)
        self.bmu_counts = bmu_counts.flatten()
        self.allowed_nodes = self.grid[bmu_counts >= self.minNumberPerBmu]
        return self.allowed_nodes

    def evaluate(self, evaluationData):
        """
        This function maps the evaluation data to the previously fitted network. It calculates the
        anomaly measure based on the distance between the observation and the K-NN nodes of this
        observation.
        :param evaluationData: Numpy array of the data to be evaluated
        :return: 1D-array with for each observation an anomaly measure
        """
        try:
            self.allowed_nodes
            assert self.allowed_nodes.shape[0] > 1
        except NameError:
            raise Exception(
                "Make sure the method fit is called before evaluating data."
            )
        except AssertionError:
            raise Exception(
                "There are no nodes satisfying the minimum criterium, algorithm cannot proceed."
            )
        else:
            classifier = NearestNeighbors(n_neighbors=self.numberOfNeighbors)
            classifier.fit(self.allowed_nodes)
            dist, _ = classifier.kneighbors(evaluationData)
        return dist.mean(axis=1)