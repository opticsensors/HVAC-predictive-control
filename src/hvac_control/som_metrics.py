import numpy as np
import numpy as np
import pandas as pd
from hvac_control.utils import argmin_first_two_axes

class SOMmetrics():
    """
    SOMmetrics class for analyzing and evaluating Self-Organizing Maps (SOMs).

    """
    def __init__(
                self,
                som,
                ):
        """
        Parameters:
        - som (numpy.ndarray): The Self-Organizing Map.

        Attributes:
        - som (numpy.ndarray): The Self-Organizing Map.
        - som_grid_size (tuple): The grid size of the SOM.
        - som_size (tuple): The size of the SOM.
        """

        self.som = som
        self.som_grid_size = (som.shape[0], som.shape[1])
        self.som_size = som.shape

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
    
    def find_two_bmu(self, X_row, som):
        """
        Find the indices of the two Best Matching Units (BMUs) for the input vector X_row.

        Parameters:
        - X_row (numpy.ndarray): Input vector.
        - som (numpy.ndarray): Self-Organizing Map.

        Returns:
        - tuple: Coordinates of the first BMU, Coordinates of the second BMU.
        """
        # min coordinates are the same for distance and the square of distance 
        distances = np.square(som - X_row).sum(axis=2)
        bmu1 = np.unravel_index(distances.argmin().astype(int), self.som_grid_size)
        distances[bmu1] = np.inf  # Set the distance of BMU1 to infinity to find BMU2
        bmu2 = np.unravel_index(distances.argmin().astype(int), self.som_grid_size)
        
        return bmu1, bmu2
    
    def find_bmu_broadcasted(self, X, som):
        #generate a distance map for every X row (every third dim belongs to a row)
        som_bro = som[:,:,:,np.newaxis]
        data_bro = X.T[np.newaxis,np.newaxis, :, :]
        distance_map = np.linalg.norm(som_bro-data_bro, axis=2)
        # find the min distance indices for every third dim 
        return argmin_first_two_axes(distance_map)

    def find_bmu_counts(self, X, som):
        """
        Map a training set to the fitted network and evaluate the number of hits for each node in the SOM.

        Parameters:
        - X (numpy.ndarray): Training set.
        - som (numpy.ndarray): Self-Organizing Map.

        Returns:
        - numpy.ndarray: Number of hits for each BMU in the SOM.
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

    def quantization_error(self, X, som):
        """
        Compute the Quantization Error for a SOM given a sample or a set of samples.

        Parameters:
        - X (numpy.ndarray): Input samples.
        - som (numpy.ndarray): Self-Organizing Map.

        Returns:
        - float: Quantization Error.
        """
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
    
    def topographic_error(self, X, som): #TODO
        """
        Compute the Topographical Error for a SOM given a sample or a set of samples.

        Parameters:
        - X (numpy.ndarray): Input samples.
        - som (numpy.ndarray): Self-Organizing Map.

        Returns:
        - float: Topographical Error.
        """
        X = np.atleast_2d(X)
        total_error = 0
        for sample in X:
            bmu1, bmu2 = self.find_two_bmu(sample, som)
            # Check if BMU1 and BMU2 are not adjacent
            if not np.all(np.abs(np.array(bmu1) - np.array(bmu2)) <= 1):
                total_error += 1
        # Compute the Topographical Error
        quantization_err = total_error / X.shape[0]
        return quantization_err

    def find_dmin(self, X, som):
        """
        Map a set of samples to the neurons of the SOM and find the minimum distance for each sample.

        Parameters:
        - X (numpy.ndarray): Input samples.
        - som (numpy.ndarray): Self-Organizing Map.

        Returns:
        - list: List of minimum distances for each sample.
        """
        X = np.atleast_2d(X)
        l_dmin = []

        for row in X:
            distance_map = np.linalg.norm(som-row, axis=2)
            l_dmin.append(distance_map.min())
    
        return l_dmin
    
    def find_bmu_and_dmin(self, X, som):
        """
        Map a set of samples to the neurons of the SOM and find the coordinates of their BMU
        along with the minimum distance associated with the BMU.

        Parameters:
        - X (numpy.ndarray): Input samples.
        - som (numpy.ndarray): Self-Organizing Map.

        Returns:
        - pd.DataFrame: DataFrame with columns 'row' (BMU row coordinate), 
                        'col' (BMU column coordinate), and 'dmin' (minimum distance).
        """
        X = np.atleast_2d(X)
        l_dmin = []
        l_coord = []

        for row in X:
            distance_map = np.linalg.norm(som-row, axis=2)
            l_dmin.append(distance_map.min())
            l_coord.append(np.unravel_index(distance_map.argmin().astype(int), self.som_grid_size))
        
        arr_coord=np.array(l_coord)
        data = {'row': arr_coord[:,0], 'col': arr_coord[:,1], 'dmin': l_dmin}
        df = pd.DataFrame(data)
        return df

    def find_dmin_min_and_max_by_units(self, df):
        """
        Find the minimum and maximum distances associated with hits for each neuron in the SOM.

        Parameters:
        - df (pd.DataFrame): DataFrame with columns 'row', 'col', and 'dmin'.

        Returns:
        - tuple: Numpy arrays representing the min and max distance maps.
        """
        df_groupby_coord = df.groupby(['row', 'col'])['dmin'].agg([('Min' , 'min'), ('Max', 'max')])
        df_groupby_coord = df_groupby_coord.sort_values(by=['row', 'col'], ascending=True).reset_index()

        dmin_map = np.full(self.som_grid_size, np.nan)
        dmax_map = np.full(self.som_grid_size, np.nan)
        dmin_map[df_groupby_coord['row'], df_groupby_coord['col']] = df_groupby_coord['Min']
        dmax_map[df_groupby_coord['row'], df_groupby_coord['col']] = df_groupby_coord['Max']

        return dmin_map, dmax_map