import os
import pandas as pd
from hvac_control.data import load_data
from hvac_control.preprocessing import *
from hvac_control.preprocessing import std_scaler_given_parameters
from hvac_control.parameters import REPO_PATH
from hvac_control.anfis import ANFIS

data_to_load = "gaia_data_1.csv"
data = load_data("gaia_data_1.csv", data_type='processed')

x_columns = ['T_ext', 'Solar_irrad', 'T_imp', 
           'BC1_power', 'BC2_power', 'Diff_temp',
           'Hours_sin', 'Hours_cos', 'T_ret'] 

y_column = ['T_ret_in_1h']

X = data[x_columns]
y = data[y_column]

# we don't split the data here because we want to train with all the data

mu_x = X.mean(0)
s_x = X.std(0)

mu_y = y.mean(0)
s_y = y.std(0)

X_norm = std_scaler_given_parameters(X, mu_x, s_x)
y_norm = std_scaler_given_parameters(y, mu_y, s_y)

n_inputs = X_norm.shape[1]
n_rules = 16
lr = 1e-4
epochs = 150
batch_size = 32

fis = ANFIS(n_inputs=n_inputs, n_rules=n_rules, learning_rate=lr, mf='gbellmf',defuzz_method='linear', loss_fun='huber', init_method='normal')
fis.compile(run_eagerly=True)

# For training
fis.fit(X_norm, y_norm, epochs=epochs, batch_size=batch_size)

path_to_save=os.path.join(REPO_PATH, 'models', 'anfis.tf')
fis.save(path_to_save)

