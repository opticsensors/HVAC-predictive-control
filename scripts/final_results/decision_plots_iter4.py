from novelty_detection.data import load_data, save_img, save_data
from novelty_detection.preprocessing import *
from novelty_detection.decision_plots import *

#load preprocessed data
data_to_load = "gaia_data_1.csv"

x_columns = ['T_ext', 'Solar_irrad', 'T_imp', 
           'BC1_power', 'BC2_power', 
           'T_ret',
           'Diff_temp', ]

y_column = ['T_ret_in_1h']

all_columns = x_columns + y_column

df_index = load_data(data_to_load, header_names=None, index=True)
df_index = filter_signal(df_index, ['T_ext', 'Solar_irrad', 'T_imp', 'BC1_power', 'BC2_power', 'T_ret'], 5)
df_index['Diff_temp'] = df_index['T_imp'] - df_index['T_ret']
df_index['T_ret_in_1h'] = df_index['T_ret'].shift(-30) # 1 hour is 30 rows, since between rows there is a 2 min interval
df_index = df_index.iloc[:-30]
df_index = remove_specific_day(df_index, '2022-05-25')
dfs_day_working_hours = remove_non_working_hours(df_index, strating_hour='05:00', ending_hour='17:30')
l_data1 = []
l_data2 = []
l_data3 = []

for day_num,df_day in enumerate(dfs_day_working_hours):
    print(f'doing plots for day num {day_num} of df...')
    df_timeseries = compute_timeseries(df_day)
    df_correlations=compute_correlations(df_day, x_columns, y_column ,max_shift=150, circular_shift=True)
    df_frequencies=compute_frequencies(df_day, all_columns, frate=1/120, max_freq=0.003)

    l_data1.append(df_timeseries)
    l_data2.append(df_correlations)
    l_data3.append(df_frequencies)

    plt1 = timeseries_plot(df_timeseries, all_columns, time='hours', main_title=f'Time series plot working hours day {day_num}', grid_size=(3,3))
    plt2 = correlation_plot(df_correlations, x_columns, main_title=f'Correlation plot working hours day {day_num}')
    plt3 = frequency_plot(df_frequencies, all_columns, main_title=f'Frecuency plot working hours day {day_num}', grid_size=(3,3))

    save_img(plt1, f'timeseries_plot_day_{day_num}.png', data_type='plots')
    save_img(plt2, f'correlation_plot_day_{day_num}.png', data_type='plots')
    save_img(plt3, f'frequency_plot_day_{day_num}.png', data_type='plots')

df_timeseries = pd.concat(l_data1).groupby(level=0).mean()
df_correlations = pd.concat(l_data2).groupby(level=0).mean()
df_frequencies = pd.concat(l_data3).groupby(level=0).mean()

plt1 = timeseries_plot(df_timeseries, all_columns, time='hours',main_title=f'Time series plot working hours average', grid_size=(3,3))
plt2 = correlation_plot(df_correlations, x_columns, main_title=f'Correlation plot working hours average')
plt3 = frequency_plot(df_frequencies, all_columns, main_title=f'Frecuency plot working hours average',grid_size=(3,3))

save_img(plt1, f'timeseries_plot_day_avg.png', data_type='plots')
save_img(plt2, f'correlation_plot_day_avg.png', data_type='plots')
save_img(plt3, f'frequency_plot_day_avg.png', data_type='plots')

