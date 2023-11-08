# Prova a partir de towardsdatascience
import numpy as np
from numpy.ma.core import ceil
from scipy.spatial import distance #distance calculation
from sklearn.preprocessing import MinMaxScaler #normalisation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #scoring
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from matplotlib import animation, colors

#python -m venv venv
#\venv\Scripts\activate
#pip install -r requirements.txt

# banknote authentication Data Set
# https://archive.ics.uci.edu/ml/datasets/banknote+authentication
# Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. 
# Irvine, CA: University of California, School of Information and Computer Science.

data_file = "data_banknote_authentication.txt"
data_x = np.loadtxt(data_file, delimiter=",", skiprows=0, usecols=range(0,4) ,dtype=np.float64)
data_y = np.loadtxt(data_file, delimiter=",", skiprows=0, usecols=(4,),dtype=np.int64)

# train and test split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape) # check the shapes

# Helper functions

# Data Normalisation
def minmax_scaler(data):
  scaler = MinMaxScaler()
  scaled = scaler.fit_transform(data)
  return scaled

# Euclidean distance
def e_distance(x,y):
  return distance.euclidean(x,y)

# Manhattan distance
def m_distance(x,y):
  return distance.cityblock(x,y)

# Best Matching Unit search
def winning_neuron(data, t, som, num_rows, num_cols):
  winner = [0,0]
  shortest_distance = np.sqrt(data.shape[1]) # initialise with max distance
  input_data = data[t]
  for row in range(num_rows):
    for col in range(num_cols):
      distance = e_distance(som[row][col], data[t])
      if distance < shortest_distance: 
        shortest_distance = distance
        winner = [row,col]
  return winner
import numpy as np

def quantization_error(data, som, num_rows, num_cols):
    total_error = 0.0
    for t in range(data.shape[0]):
        winner = winning_neuron(data, t, som, num_rows, num_cols)
        total_error += e_distance(data[t], som[winner[0]][winner[1]])
    quantization_err = total_error / data.shape[0]
    return quantization_err

def printplot(train_y, num_rows, num_cols, train_x_norm, som, max_steps):
# collecting labels
  label_data = train_y
  map = np.empty(shape=(num_rows, num_cols), dtype=object)

  for row in range(num_rows):
    for col in range(num_cols):
      map[row][col] = [] # empty list to store the label

  for t in range(train_x_norm.shape[0]):
    if (t+1) % 1000 == 0:
      print("sample data: ", t+1)
    winner = winning_neuron(train_x_norm, t, som, num_rows, num_cols)
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

  title = ('Iteration ' + str(max_steps))
  cmap = colors.ListedColormap(['tab:green', 'tab:red', 'tab:orange'])
  plt.imshow(label_map, cmap=cmap)
  plt.colorbar()
  plt.title(title)
  plt.show()
  return 0

# Learning rate and neighbourhood range calculation
def decay(step, max_steps,max_learning_rate,max_m_dsitance):
  coefficient = 1.0 - (np.float64(step)/max_steps)
  learning_rate = coefficient*max_learning_rate
  neighbourhood_range = ceil(coefficient * max_m_dsitance)
  return learning_rate, neighbourhood_range

# hyperparameters
num_rows = 10
num_cols = 10
max_m_dsitance = 4
max_learning_rate = 0.5
max_steps = int(7*10e3) # Multiplies by 10

# num_nurons = 5*np.sqrt(train_x.shape[0])
# grid_size = ceil(np.sqrt(num_nurons))
# print(grid_size)

#main function

train_x_norm = minmax_scaler(train_x) # normalisation

# initialising self-organising map
num_dims = train_x_norm.shape[1] # numnber of dimensions in the input data
np.random.seed(40)
som = np.random.random_sample(size=(num_rows, num_cols, num_dims)) # random map construction

# start training iterations
for step in range(max_steps):
  if (step+1) % 1000 == 0:
    #printplot(train_y, num_rows, num_cols, train_x_norm, som, max_steps)
    print("Iteration: ", step+1) # print out the current iteration for every 1k
  learning_rate, neighbourhood_range = decay(step, max_steps,max_learning_rate,max_m_dsitance)

  t = np.random.randint(0,high=train_x_norm.shape[0]) # random index of traing data
  winner = winning_neuron(train_x_norm, t, som, num_rows, num_cols)
  for row in range(num_rows):
    for col in range(num_cols):
      if m_distance([row,col],winner) <= neighbourhood_range:
        som[row][col] += learning_rate*(train_x_norm[t]-som[row][col]) #update neighbour's weight

print("SOM training completed")
# collecting labels
label_data = train_y
map = np.empty(shape=(num_rows, num_cols), dtype=object)

for row in range(num_rows):
  for col in range(num_cols):
    map[row][col] = [] # empty list to store the label

for t in range(train_x_norm.shape[0]):
  if (t+1) % 1000 == 0:
    print("sample data: ", t+1)
  winner = winning_neuron(train_x_norm, t, som, num_rows, num_cols)
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

title = ('Iteration ' + str(max_steps))
cmap = colors.ListedColormap(['tab:green', 'tab:red', 'tab:orange'])
plt.imshow(label_map, cmap=cmap)
plt.colorbar()
plt.title(title)
plt.show()
# Predicting the test set labels. Test data

# using the trained som, search the winning node of corresponding to the test data
# get the label of the winning node

data = minmax_scaler(test_x) # normalisation

winner_labels = []

for t in range(data.shape[0]):
 winner = winning_neuron(data, t, som, num_rows, num_cols)
 row = winner[0]
 col = winner[1]
 predicted = label_map[row][col]
 winner_labels.append(predicted)

print("Accuracy: ",accuracy_score(test_y, np.array(winner_labels)))

conf_matrix = confusion_matrix(test_y, np.array(winner_labels))

# extra data

TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

cm_accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Confusion Matrix: ", conf_matrix)
#ConfusionMatrixDisplay.from_predictions(test_y, np.array(winner_labels))
#plt.show()
# analysis of the confusion matrix

print("Average precission score: ",average_precision_score(test_y, np.array(winner_labels)))

quantization_err = quantization_error(train_x_norm, som, num_rows, num_cols)
print("Quantization Error:", quantization_err)


# input analysis
# data summary statistics
mean_values = np.mean(data_x, axis=0)
median_values = np.median(data_x, axis=0)
std_deviation = np.std(data_x, axis=0)
min_values = np.min(data_x, axis=0)
max_values = np.max(data_x, axis=0)
print("Mean: ", mean_values)
print("Median: ", median_values)
print("Standard deviation: ", std_deviation)

# data distribution visualization
import matplotlib.pyplot as plt

# Example: Histogram for the first feature
plt.hist(data_x[:, 0], bins=30)
plt.xlabel("Feature 1")
plt.ylabel("Frequency")
plt.title("Histogram of Feature 1")
plt.show()

# class distribution
class_labels, class_counts = np.unique(data_y, return_counts=True)

# correlation analysis
correlation_matrix = np.corrcoef(data_x, rowvar=False)


# More evaluation methods: https://scikit-learn.org/stable/modules/model_evaluation.html
#    Computes the quantization error of the SOM.
#    It uses the data fed at last training.
#    The neuron with the shortest distance from the input signal becomes the BMU.
