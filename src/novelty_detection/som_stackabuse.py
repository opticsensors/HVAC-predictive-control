import numpy as np

class SOM_stackabuse():
  """
  Adapted from: https://stackabuse.com/self-organizing-maps-theory-and-implementation-in-python-with-numpy/
  """
  def __init__(self,
              step=3,
              learn_rate = .1, 
              radius_sq = 1, 
              lr_decay = .1, 
              radius_decay = .1, 
              epochs = 10,
              random_state=40): 
       
    self.step = step
    self.learn_rate = learn_rate
    self.radius_sq = radius_sq
    self.lr_decay = lr_decay
    self.radius_decay = radius_decay
    self.epochs = epochs
    self.random_state = random_state

  def find_BMU(self, som, x):
    """
    Best Matching Unit search. Return the (g,h) index of the BMU in the grid
    """
    distSq = (np.square(som - x)).sum(axis=2)
    return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)
      

  def update_weights(self, som, train_ex, BMU_coord):
      """
      Update the weights of the SOM cells when given a single training example
      and the model parameters along with BMU coordinates as a tuple
      """
      g, h = BMU_coord
      #if radius is close to zero then only BMU is changed
      if self.radius_sq < 1e-3:
          som[g,h,:] += self.learn_rate * (train_ex - som[g,h,:])
          return som
      # Change all cells in a small neighborhood of BMU
      for i in range(max(0, g-self.step), min(som.shape[0], g+self.step)):
          for j in range(max(0, h-self.step), min(som.shape[1], h+self.step)):
              dist_sq = np.square(i - g) + np.square(j - h)
              dist_func = np.exp(-dist_sq / 2 / self.radius_sq)
              som[i,j,:] += self.learn_rate * dist_func * (train_ex - som[i,j,:])   
      return som    


  def fit(self, som, train_data):    
    """
    Main routine for training an SOM. It requires an initialized SOM grid
    or a partially trained grid as parameter
    """

    learn_rate_0 = self.learn_rate
    radius_0 = self.radius_sq
    rand = np.random.RandomState(0)
    for epoch in np.arange(0, self.epochs):
        rand.shuffle(train_data)      
        for train_ex in train_data:
            g, h = self.find_BMU(som, train_ex)
            som = self.update_weights(som, train_ex, (g,h))
        # Update learning rate and radius
        self.learn_rate = learn_rate_0 * np.exp(-epoch * self.lr_decay)
        self.radius_sq = radius_0 * np.exp(-epoch * self.radius_decay)            
    return som
