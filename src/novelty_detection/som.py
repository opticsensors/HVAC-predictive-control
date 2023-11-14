import numpy as np
from sklearn.metrics import (accuracy_score,
                            confusion_matrix,
                            average_precision_score)
from numpy.ma.core import ceil
from sklearn.utils import check_array, check_X_y
from novelty_detection.utils import (argmin_first_two_axes,
                                     choose_random_sample_numpy,
                                     choose_random_array_numpy)


class SOM():

  def __init__(self,
              som_grid_size=(10,10),
              max_cityblock=4,
              max_learning_rate=0.5,
              max_steps=70000,
              random_state=40
              ):

    self.som_grid_size = som_grid_size
    self.max_cityblock = max_cityblock
    self.max_learning_rate = max_learning_rate
    self.max_steps = max_steps
    self.random_state = random_state
    np.random.seed(random_state) #TODO sklearn does that internally?

  def get_params(self, deep=True):
    # suppose this estimator has parameters "alpha" and "recursive"
    return {"som_grid_size": self.som_grid_size, 
            "max_cityblock": self.max_cityblock,
            "max_learning_rate": self.max_learning_rate, 
            "max_learning_rate": self.max_learning_rate, 
            "random_state": self.random_state,
            }

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self

  def winning_neuron(self, data_row, som):
    """
    Best Matching Unit search
    """
    distance_map = np.sqrt(np.sum((som-data_row)**2, axis=2))
    return np.unravel_index(np.nanargmin(distance_map, axis=None), distance_map.shape)

  def winning_neuron_broadcasted(self, data, som):
    """
    Best Matching Unit search using broadcasting
    """
    #generate a distance map for every X row (every third dim belongs to a row)
    som_bro = som[:,:,:,np.newaxis]
    data_bro = data.T[np.newaxis,np.newaxis, :, :]
    distance_map = np.sqrt(np.sum((som_bro-data_bro)**2, axis=2))
    # find the min distance indices for every third dim 
    return argmin_first_two_axes(distance_map)

  def decay(self, step):
    """
    Learning rate and neighbourhood range calculation
    """
    coefficient = 1.0 - (np.float64(step)/self.max_steps)
    # linear decent
    learning_rate = coefficient*self.max_learning_rate
    # reduces by half every few steps
    neighbourhood_range = ceil(coefficient * self.max_cityblock)
    return learning_rate, neighbourhood_range

  def quantization_error(self, data, som):
    total_error = 0.0
    for row in data:
        distance_map = np.sqrt(np.sum((som-row)**2, axis=2))
        total_error += distance_map.min()
    quantization_err = total_error / data.shape[0]
    return quantization_err

  def quantization_error_broadcasted(self, data, som):
    som_bro = som[:,:,:,np.newaxis]
    data_bro = data.T[np.newaxis,np.newaxis, :, :]
    distance_map = np.sqrt(np.sum((som_bro-data_bro)**2, axis=2))
    total_error = np.min(distance_map, axis=(0,1)).sum()
    quantization_err = total_error / data.shape[0]
    return quantization_err
  
  def fit(self, X, y):
    # Check that X and y have correct shape
    X, y = check_X_y(X, y)
    # Store the classes seen during fit
    self.classes_ = np.unique(y)
    self.X_ = X
    self.y_ = y
    self.som_size = (self.som_grid_size[0], self.som_grid_size[1], X.shape[1])
    som = choose_random_array_numpy(self.som_grid_size)
    indices_map = np.array(list(np.ndindex(self.som_grid_size[0], self.som_grid_size[1])))

    for step in range(self.max_steps):
      if (step+1) % 1000 == 0:
      #printplot(train_y, num_rows, num_cols, train_x_norm, som, max_steps)
          print("Iteration: ", step+1) # print out the current iteration for every 1k

      learning_rate, neighbourhood_range = self.decay(step)
      X_row = choose_random_sample_numpy(X)
      winner = self.winning_neuron(X_row, som)
      winner_arr = np.array([winner])
      cityblock_map = np.abs(winner_arr[:,None] - indices_map).sum(-1).reshape(self.som_size[0],self.som_size[1])
      neighbourhood_range_cond = (cityblock_map<=neighbourhood_range)
      som[neighbourhood_range_cond] += learning_rate*(X_row-som[neighbourhood_range_cond])

    self.som_ = som
    print("SOM training completed")

    idx = self.winning_neuron_broadcasted(X,som)
    idx_with_y_value = np.concatenate([idx, y.reshape(-1,1)], axis=1)
    idx_with_ones = idx_with_y_value[idx_with_y_value[:,2]==1]

    # find the amount of times min distance indices are repeated among third dim
    unique_idx, counts_idx = np.unique(idx, return_counts=True, axis=0)

    # find the amount of times min distance indices with an associated y value of 1 are repeated among third dim
    unique_idx_with_ones, counts_idx_with_ones = np.unique(idx_with_ones, return_counts=True, axis=0)

    occurence_freq_map = np.zeros(shape=self.som_grid_size)
    ones_freq_map = np.zeros(shape=self.som_grid_size)
    label_map = np.zeros(shape=self.som_grid_size)
    occurence_freq_map[unique_idx[:,0], unique_idx[:,1]] = counts_idx
    ones_freq_map[unique_idx_with_ones[:,0], unique_idx_with_ones[:,1]] = counts_idx_with_ones

    label_map = ones_freq_map/occurence_freq_map
    classification_cond = (label_map>0.5)
    label_map[classification_cond] = 1
    label_map[~classification_cond] = 0
    label_map[occurence_freq_map==0] = 2

    self.label_map_ = label_map
    # Return the model
    return self

  def predict(self, X):
    X = check_array(X)
    idx = self.winning_neuron_broadcasted(X, self.som_)
    y_pred = self.label_map_[idx[:,0], idx[:,1]]
    return y_pred

  def score(self, y_pred, test_y):
    acc = accuracy_score(test_y, y_pred)
    conf_matrix = confusion_matrix(test_y, y_pred)
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    cm_accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    avg_precision = average_precision_score(test_y, y_pred)
    quantization_err = self.quantization_error(self.X_, self.som_)   
        
    # input analysis
    # data summary statistics

    # mean_values = np.mean(data_x, axis=0)
    # median_values = np.median(data_x, axis=0)
    # std_deviation = np.std(data_x, axis=0)
    # min_values = np.min(data_x, axis=0)
    # max_values = np.max(data_x, axis=0)

    # class distribution
    # class_labels, class_counts = np.unique(data_y, return_counts=True)

    # correlation analysis
    # correlation_matrix = np.corrcoef(data_x, rowvar=False)





