from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from novelty_detection.data import load_data, save_img
from novelty_detection.preprocessing import *
from novelty_detection.preprocessing import minmax_scaler_given_parameters, std_scaler_given_parameters
from novelty_detection.som import SOM

#load data and preprocessing
df = load_data("gaia_data.csv", header_names=None)

columns = ['T_ext', 'Solar_irrad', 'T_imp', 
           'BC1_power', 'BC2_power', 'Refr1_power', 
           'Refr2_power', 'BC1_flow', 'BC2_flow', 
           'Refr1_flow', 'Refr2_flow', 'T_ret']

max_minutes=30
rows_to_skip=2
max_consecutive_nans=30
thresh_len=1000

dfs = preprocessing_pipeline(df, max_minutes, rows_to_skip, max_consecutive_nans, thresh_len, columns)
choosen_df=1
df_index = dfs[choosen_df]


# decision plots
x_columns = ['T_ext', 'Solar_irrad', 'T_imp', 
           'BC1_power', 'BC2_power', 'Refr1_power', 
           'Refr2_power', 'BC1_flow', 'BC2_flow', 
           'Refr1_flow', 'Refr2_flow']

y_column = ['T_ret']

all_columns = x_columns + y_column

# plt1 = timeseries_plot(df_index, all_columns, (3,4), plot_size=(500,800), margin=300, spacing =435, dpi=200.)
# plt2 = correlation_plot(df_index, x_columns, max_shift=2000, plot_size=(600,1000), margin=300, spacing =435, dpi=200.)
# plt3 = frequency_plot(df_index, all_columns, (3,4), frate=1/120, max_freq=0.00004, plot_size=(500,800), margin=150, spacing =300, dpi=200.)
# 
# save_img(plt1, 'timeseries_plot.png', data_type='plots')
# save_img(plt2, 'correlation_plot.png', data_type='plots')
# save_img(plt3, 'frequency_plot.png', data_type='plots')


# train and test split
X=df_index[x_columns].to_numpy()
y=df_index[y_column].to_numpy()

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
print('x train shape: ', train_x.shape)
print('y train shape: ', train_y.shape) 
print('x test shape: ', test_x.shape)
print('x test shape: ', test_y.shape)


#Scale
max_val = train_x.max()
min_val = train_x.min()
mu = train_x.mean(0)
s = train_x.std(0)

#train_x_norm = minmax_scaler_given_parameters(train_x, min_val, max_val)
#test_x_norm = minmax_scaler_given_parameters(test_x, min_val, max_val)

train_x_norm = std_scaler_given_parameters(train_x, mu, s)
test_x_norm = std_scaler_given_parameters(test_x, mu, s)


# Train SOM
model=SOM(som_grid_size=(20,20),
          max_distance=4,
          learning_rate_0=0.5,
          max_iter=100000,
          random_state=0,
          sigma_0=1, 
          sigma_decay=0.0005,
          learning_rate_decay=0.0005,
          methods={'init_som': 'sq_normal',
                  'bmu_distance': 'euclidian',
                  'compute_neighborhood': 'exp',
                  'update_sigma_and_learning_rate': 'exp'}) 
model=model.fit(train_x_norm, epochs=8)
som=model.som
som_dataset=som.reshape(-1,som.shape[2])


# PCA vizualization
pca = PCA(n_components = 2)
pca_som = pca.fit_transform(som_dataset)

exp_variance_2d = pca.explained_variance_ratio_
print(f"SOM data 2D: Total = {np.sum(exp_variance_2d)} and per components = {exp_variance_2d}")

pca_train = pca.fit_transform(train_x_norm)
exp_variance_2d = pca.explained_variance_ratio_
print(f"Train data 2D: Total = {np.sum(exp_variance_2d)} and per components = {exp_variance_2d}")

fig, ax = plt.subplots()
ax.scatter(pca_train[:,0], pca_train[:,1], color="red", label="train data", edgecolors='none')
ax.scatter(pca_som[:,0], pca_som[:,1], color="orange", label="som data", edgecolors='none')
ax.legend()
ax.grid(True)
fig.suptitle(f"2D PCA with exp variance = {round(np.sum(exp_variance_2d), 3)}")
plt.show()