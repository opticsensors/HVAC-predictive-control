from novelty_detection.data import load_data, save_img, save_data
from novelty_detection.preprocessing import *
from novelty_detection.decision_plots import *

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

dfs = preprocessing_pipeline(df, max_minutes, rows_to_skip, max_consecutive_nans, thresh_len, all_columns)

for num,df in enumerate(dfs):
    print(f'saving df {num}')
    name = f'gaia_data_{num}.csv'
    save_data(df, name, data_type='processed', index=True)
