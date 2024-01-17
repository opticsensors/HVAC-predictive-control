from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from novelty_detection.data import load_data, save_img, save_data
from novelty_detection.preprocessing import *
from novelty_detection.preprocessing import minmax_scaler_given_parameters, std_scaler_given_parameters
from novelty_detection.som import SOM

#load data and preprocessing
df = load_data("gaia_data.csv", header_names=None)

x_columns = ['T_ext', 'Solar_irrad', 'T_imp', 
           'BC1_power', 'BC2_power', 'Refr1_power', 
           'Refr2_power', 'BC1_flow', 'BC2_flow', 
           'Refr1_flow', 'Refr2_flow']

y_column = ['T_ret']

all_columns = x_columns + y_column

max_minutes=30
rows_to_skip=2
max_consecutive_nans=30
thresh_len=1000
choosen_df=1

dfs = preprocessing_pipeline(df, max_minutes, rows_to_skip, max_consecutive_nans, thresh_len, all_columns)

for num,df in enumerate(dfs):
    print(f'saving df {num}')
    name = f'gaia_data_{num}.csv'
    save_data(df, name, data_type='processed', index=True)

df_index = dfs[choosen_df]

df_index = remove_specific_day(df_index, '2022-05-25')
dfs_day_working_hours = remove_non_working_hours(df_index, strating_hour='04:55', ending_hour='18:30')
l_data1 = []
l_data2 = []
l_data3 = []

for day_num,df_day in enumerate(dfs_day_working_hours):
    print(f'doing plots for day num {day_num} of df...')
    l_data1.append(compute_timeseries(df_day))
    l_data2.append(compute_correlations(df_day, x_columns, max_shift=90, circular_shift=True))
    l_data3.append(compute_frequencies(df_day, all_columns, frate=1/120, max_freq=0.0006))

df_timeseries = pd.concat(l_data1).groupby(level=0).mean()
df_correlations = pd.concat(l_data2).groupby(level=0).mean()
df_frequencies = pd.concat(l_data3).groupby(level=0).mean()

plt1 = timeseries_plot(df_timeseries, all_columns, time='hours',main_title=f'Time series plot working hours average')
plt2 = correlation_plot(df_correlations, x_columns, main_title=f'Correlation plot working hours average')
plt3 = frequency_plot(df_frequencies, all_columns, main_title=f'Frecuency plot working hours average')

save_img(plt1, f'timeseries_plot_day_avg.png', data_type='plots')
save_img(plt2, f'correlation_plot_day_avg.png', data_type='plots')
save_img(plt3, f'frequency_plot_day_avg.png', data_type='plots')


for day_num,df_day in enumerate(dfs_day_working_hours):
    print(f'doing plots again for day num {day_num} of df...')
    df_timeseries = compute_timeseries(df_day)
    df_correlations=compute_correlations(df_day, x_columns, max_shift=90, circular_shift=True)
    df_frequencies=compute_frequencies(df_day, all_columns, frate=1/120, max_freq=0.0006)

    plt1 = timeseries_plot(df_timeseries, all_columns, time='hours', main_title=f'Time series plot working hours day {day_num}')
    plt2 = correlation_plot(df_correlations, x_columns, main_title=f'Correlation plot working hours day {day_num}')
    plt3 = frequency_plot(df_frequencies, all_columns, main_title=f'Frecuency plot working hours day {day_num}')

    save_img(plt1, f'timeseries_plot_day_{day_num}.png', data_type='plots')
    save_img(plt2, f'correlation_plot_day_{day_num}.png', data_type='plots')
    save_img(plt3, f'frequency_plot_day_{day_num}.png', data_type='plots')




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