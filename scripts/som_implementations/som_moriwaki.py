import numpy as np
from numpy.ma.core import ceil
from sklearn.utils import check_array, check_X_y
from hvac_control.utils import (argmin_first_two_axes,
                                     choose_random_sample,
                                     choose_random_array)
from scipy.spatial import distance 

class SOM_vectorized():
  """
  Vectorized for better performence than moriwaki implementation
  """

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
    som = choose_random_array(self.som_size)
    indices_map = np.array(list(np.ndindex(self.som_grid_size[0], self.som_grid_size[1])))

    for step in range(self.max_steps):
      if (step+1) % 1000 == 0:
          print("Iteration: ", step+1) # print out the current iteration for every 1k

      learning_rate, neighbourhood_range = self.decay(step)
      X_row = choose_random_sample(X)
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


class SOM_original(SOM_vectorized):
  """
  Adapted from: https://towardsdatascience.com/understanding-self-organising-map-neural-network-with-python-code-7a77f501e985
  """
  def __init__(self,
              som_grid_size=(10,10),
              max_cityblock=4,
              max_learning_rate=0.5,
              max_steps=70000,
              random_state=40
              ):
    super().__init__(som_grid_size, max_cityblock, max_learning_rate, max_steps, random_state)

  def winning_neuron(self, data, t, som):
    """
    Best Matching Unit search
    """
    winner = [0,0]
    shortest_distance = np.sqrt(data.shape[1]) # initialise with max distance
    num_rows=self.som_grid_size[0]
    num_cols=self.som_grid_size[1]
    for row in range(num_rows):
      for col in range(num_cols):
        d = distance.euclidean(som[row][col], data[t])
        if d < shortest_distance: 
          shortest_distance = d
          winner = [row,col]
    return winner

  def quantization_error(self, data, som):
    total_error = 0.0
    for t in range(data.shape[0]):
        winner = self.winning_neuron(data, t, som)
        total_error += distance.euclidean(data[t], som[winner[0]][winner[1]])
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
    num_rows=self.som_grid_size[0]
    num_cols=self.som_grid_size[1]
    num_dims=X.shape[1]
    som = choose_random_array(self.som_size)

    # start training iterations
    for step in range(self.max_steps):
      if (step+1) % 1000 == 0:
        #printplot(train_y, num_rows, num_cols, train_x_norm, som, max_steps)
        print("Iteration: ", step+1) # print out the current iteration for every 1k
      learning_rate, neighbourhood_range = self.decay(step)

      t = np.random.randint(0,high=X.shape[0]) # random index of traing data
      winner = self.winning_neuron(X, t, som)
      for row in range(num_rows):
        for col in range(num_cols):
          if distance.cityblock([row,col],winner) <= neighbourhood_range:
            som[row][col] += learning_rate*(X[t]-som[row][col]) #update neighbour's weight

    self.som_ = som
    print("SOM training completed")

    # collecting labels
    label_data = y
    map = np.empty(shape=(num_rows, num_cols), dtype=object)

    for row in range(num_rows):
      for col in range(num_cols):
        map[row][col] = [] # empty list to store the label

    for t in range(X.shape[0]):
      if (t+1) % 1000 == 0:
        print("sample data: ", t+1)
      winner = self.winning_neuron(X, t, som)
      map[winner[0]][winner[1]].append(label_data[t]) # label of winning neuron

    # construct label map
    label_map = np.zeros(shape=(num_rows, num_cols),dtype=np.int64)
    for row in range(num_rows):
      for col in range(num_cols):
        label_list = map[row][col]
        if len(label_list)==0:
          label = 2
        else:
          label = max(label_list, key=label_list.count)
        label_map[row][col] = label

    self.label_map_ = label_map
    # Return the model
    return self

  def predict(self, X):
    X = check_array(X)
    y_pred = []

    for t in range(X.shape[0]):
      winner = self.winning_neuron(X, t, self.som_)
      row = winner[0]
      col = winner[1]
      predicted = self.label_map_[row][col]
      y_pred.append(predicted)
    return y_pred