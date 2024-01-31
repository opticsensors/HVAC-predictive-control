from sklearn.decomposition import PCA
from hvac_control.som import SOM
from sklearn.model_selection import train_test_split
from hvac_control.preprocessing import minmax_scaler_given_parameters, std_scaler_given_parameters

# train and test split
# X=df_index[x_columns].to_numpy()
# y=df_index[y_column].to_numpy()
# 
# train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
# print('x train shape: ', train_x.shape)
# print('y train shape: ', train_y.shape) 
# print('x test shape: ', test_x.shape)
# print('x test shape: ', test_y.shape)
# 
# 
# #Scale
# max_val = train_x.max()
# min_val = train_x.min()
# mu = train_x.mean(0)
# s = train_x.std(0)
# 
# #train_x_norm = minmax_scaler_given_parameters(train_x, min_val, max_val)
# #test_x_norm = minmax_scaler_given_parameters(test_x, min_val, max_val)
# 
# train_x_norm = std_scaler_given_parameters(train_x, mu, s)
# test_x_norm = std_scaler_given_parameters(test_x, mu, s)
# 
# 
# # Train SOM
# model=SOM(som_grid_size=(15,15),
#           max_distance=5,
#           learning_rate_0=0.5,
#           max_iter=100000,
#           random_state=0,
#           sigma_0=1, 
#           sigma_decay=0.0005,
#           learning_rate_decay=0.0005,
#           methods={'init_som': 'uniform',
#                   'bmu_distance': 'cityblock',
#                   'compute_neighborhood': 'ceil',
#                   'update_sigma_and_learning_rate': 'linear'}) 
# model=model.fit(train_x_norm, epochs=8)
# som=model.som
# som_dataset=som.reshape(-1,som.shape[2])
# 
# 
# # PCA vizualization
# pca = PCA(n_components = 2)
# pca_som = pca.fit_transform(som_dataset)
# 
# exp_variance_2d = pca.explained_variance_ratio_
# print(f"SOM data 2D: Total = {np.sum(exp_variance_2d)} and per components = {exp_variance_2d}")
# 
# pca_train = pca.fit_transform(train_x_norm)
# exp_variance_2d = pca.explained_variance_ratio_
# print(f"Train data 2D: Total = {np.sum(exp_variance_2d)} and per components = {exp_variance_2d}")
# 
# fig, ax = plt.subplots()
# ax.scatter(pca_train[:,0], pca_train[:,1], color="red", label="train data", edgecolors='none')
# ax.scatter(pca_som[:,0], pca_som[:,1], color="orange", label="som data", edgecolors='none')
# ax.legend()
# ax.grid(True)
# fig.suptitle(f"2D PCA with exp variance = {round(np.sum(exp_variance_2d), 3)}")
# plt.show()