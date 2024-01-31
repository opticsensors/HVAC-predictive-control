from hvac_control.data import load_data, save_img, save_data
from hvac_control.preprocessing import *
from hvac_control.decision_plots import *

#load data and preprocessing
df = load_data("gaia_data.csv", header_names=None)

# FIRST PREPROCESSING STEP
print('first preprocessing step ...')

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

dfs = first_preprocessing_step(df, max_minutes, rows_to_skip, max_consecutive_nans, thresh_len, all_columns)

for num,df in enumerate(dfs):
    print(f'saving df {num}')
    name = f'gaia_data_{num}.csv'
    save_data(df, name, data_type='interim', index=True)


# SECOND PREPROCESSING STEP
print('second preprocessing step ...')

columns_to_filter =  ['T_ext', 'Solar_irrad', 'T_imp', 'BC1_power', 'BC2_power', 'T_ret']    

x_columns = ['Day', 'T_ext', 'Solar_irrad', 'T_imp', 
           'BC1_power', 'BC2_power', 'Diff_temp',
           'Hours_sin', 'Hours_cos', 'T_ret'] 

y_column = ['T_ret_in_1h']

all_columns = x_columns + y_column

strating_hour='05:00'
ending_hour='17:30'
removed_day='2022-05-25'
kernel_size=5
projection = 30

for num,df in enumerate(dfs):
    print(f'saving df {num}')
    name = f'gaia_data_{num}.csv'
    df_processed = second_preprocessing_step(df, all_columns, columns_to_filter, strating_hour, ending_hour, removed_day, kernel_size=5, projection = 30, )
    save_data(df_processed, name, data_type='processed', index=False)
