import os
import pandas as pd
from hvac_control.data import load_data
from hvac_control.preprocessing import *
from hvac_control.preprocessing import std_scaler_given_parameters
from hvac_control.som import SOM
from hvac_control.parameters import REPO_PATH

data_to_load = "gaia_data_1.csv"
df_index = load_data(data_to_load, data_type='interim', header_names=None, index=True)

df_working = remove_specific_day(df_index, '2022-05-25')
dfs_working = remove_non_working_hours(df_working, strating_hour='05:00', ending_hour='17:30')
df_working= pd.concat(dfs_working)

x_columns = ['T_ext', 'Solar_irrad', 'T_imp', 
           'BC1_power', 'BC2_power', 'Refr1_power', 
           'BC1_flow', 'BC2_flow', 
           'Refr2_flow']

y_column = ['T_ret']

all_columns = x_columns + y_column

X=df_working[all_columns].to_numpy()

# we don't split the data here because we want to train with all the data

mu = X.mean(0)
s = X.std(0)

X_norm = std_scaler_given_parameters(X, mu, s)

model=SOM(som_grid_size=(12,12),
          max_distance=6,
          learning_rate_0=0.5,
          max_iter=100000,
          random_state=0,
          sigma_0=1, 
          sigma_decay=0.0005,
          learning_rate_decay=0.0005,
          methods={'init_som': 'uniform',
                  'bmu_distance': 'cityblock',
                  'compute_neighborhood': 'ceil',
                  'update_sigma_and_learning_rate': 'linear'}) 
model=model.fit(X_norm, epochs=8)
som=model.som
som_dataset=som.reshape(-1,som.shape[2])

path_to_save=os.path.join(REPO_PATH, 'models', 'som.npy')
np.save(path_to_save, som)
