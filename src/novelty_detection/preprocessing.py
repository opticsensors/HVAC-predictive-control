import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def minmax_scaler(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled

def std_scaler(data):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    return scaled

def minmax_scaler_given_parameters(data, max_val, min_val, feature_range=[0,1]):
    data_std = (data - min_val) / (max_val - min_val)
    scaled = data_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    return scaled

def std_scaler_given_parameters(data, mu, sigma):
    scaled = (data - mu) / sigma
    return scaled

def convert_df_time_column_to_datetime(df):

    for x in df.columns:
        if df[x].astype(str).str.match(r'\d{4}-\d{2}-\d{2} \d{2}\:\d{2}\:\d{2}').all():
            df = df.rename(columns={x: "datetime"})
            df["datetime"] = pd.to_datetime(df['datetime'])
    return df

def convert_df_to_df_with_datetime_index(df):

    if df.index.inferred_type == 'datetime64':
        return df
    else:
        return df.set_index('datetime')

def make_df_continuous_in_time(df, max_minutes=30):

    time_diff = pd.Series(df.index).diff()
    time_diff_min = time_diff.astype(np.int64) / (60*10**9)
    values, counts = np.unique(time_diff_min, return_counts=True)

    list_of_idxs=[]

    for val in values:
        if val > max_minutes:
            idxs = list(np.where(time_diff_min == val)[0])
            list_of_idxs += idxs

    list_of_idxs.sort()
    l_mod = [0] + list_of_idxs + [len(df)+1]

    dfs = [df.iloc[l_mod[n]:l_mod[n+1]] for n in range(len(l_mod)-1)]

    dfs_continuous=[]
    for i,df in enumerate(dfs):
        dfs_continuous.append(df.resample(rule='1T').mean())  

    return dfs_continuous

def convert_dfs_variables_to_same_frequency(dfs, rows_to_skip=2):

    optimal_rows = {}

    for i, e in enumerate(dfs):
        
        min_nans = np.inf  
        optimal_row = None
        
        for row_to_start in range(rows_to_skip):
            temp = e.iloc[row_to_start::rows_to_skip, :]
            total_nans = temp.isnull().sum().sum()
            if total_nans < min_nans:
                min_nans = total_nans
                optimal_row = row_to_start
        optimal_rows[i] = optimal_row

    dfs_freq_reduced=[]
    for i, optimal_row in optimal_rows.items():
        df_i=dfs[i]
        dfs_freq_reduced.append(df_i.iloc[optimal_row::rows_to_skip, :])

    return dfs_freq_reduced

def split_dfs_based_on_consecutive_nans(dfs, max_consecutive_nans):

    dfs_nan_split=[]

    for df in dfs:
        df_copy = df.copy()
        df_copy['isna'] = df_copy['T_ret'].isna()
        df_copy['Group1']=df_copy['isna'].ne(df_copy['isna'].shift()).cumsum()
        df_copy['count']=df_copy.groupby('Group1')['Group1'].transform('size')
        df_copy['invalid_rows']=(df_copy['count'] > max_consecutive_nans) & (df_copy['isna'])
        df_copy['Group2']=df_copy['invalid_rows'].ne(df_copy['invalid_rows'].shift()).cumsum()

        for _, g in df_copy.groupby(df_copy['Group2']):
            if g['invalid_rows'].all()==False:
                dfs_nan_split.append(g)

    return dfs_nan_split

def fill_dfs_nans_and_keep_long_dfs_only(dfs, thresh_len=200):

    dfs_valid = []
    for df in dfs:
        length=len(df)
        if length > thresh_len:
            dfs_valid.append(df.interpolate(method='linear'))
    
    return dfs_valid

def rearrange_and_keep_important_columns(dfs, columns):

    dfs_columns=[]
    for df in dfs:
        dfs_columns.append(df[columns])

    return dfs_columns

def preprocessing_pipeline(df, max_minutes, rows_to_skip, max_consecutive_nans, thresh_len, columns):

    df_date = convert_df_time_column_to_datetime(df)
    df_index = convert_df_to_df_with_datetime_index(df_date)
    dfs_continuous = make_df_continuous_in_time(df_index,  max_minutes)
    dfs_reduced = convert_dfs_variables_to_same_frequency(dfs_continuous, rows_to_skip)
    dfs_trim = split_dfs_based_on_consecutive_nans(dfs_reduced, max_consecutive_nans)
    dfs_valid = fill_dfs_nans_and_keep_long_dfs_only(dfs_trim, thresh_len)
    dfs_columns = rearrange_and_keep_important_columns(dfs_valid, columns)

    return dfs_columns

def remove_weekends(df):

    dayofweek = df.index.to_series().dt.dayofweek
    df['day_of_week'] = dayofweek.to_numpy()
    df['isweekday'] = df['day_of_week']<5
    df['group']=df['isweekday'].ne(df['isweekday'].shift()).cumsum()

    dfs_weekdays=[]
    for i, g in df.groupby(df['group']):
        if g['isweekday'].all()==True:
            g=g.drop(columns=['day_of_week','isweekday','group'])
            dfs_weekdays.append(g)

    return dfs_weekdays

def remove_specific_day(df, removed_day):

    return df.drop(df.loc[removed_day].index)

def remove_non_working_hours(df, strating_hour='04:32', ending_hour='18:30'):

    df = df.between_time(strating_hour,ending_hour)
    df = df[df.index.dayofweek < 5]
    dfs_day_working_hours = [group[1] for group in df.groupby(df.index.date)]
    dfs_day_working_hours = [df for df in dfs_day_working_hours if len(df) == len(max(dfs_day_working_hours, key=len))]
    
    return dfs_day_working_hours    

def filter_signal_non_causal(df, columns, kernel_size=5):
    
    df_filtered = df.copy()
    outside = kernel_size//2
    for col in columns:
        signal = df[col]
        signal_pad = np.pad(signal, (outside, outside), 'edge')
        signal_conv = np.convolve(signal_pad, np.ones((kernel_size,))/kernel_size, mode='valid')
        df_filtered[col] = signal_conv

    return df_filtered

def filter_signal_causal(df, columns, kernel_size=5):
    
    kernel_size = 5
    offset = kernel_size//2

    df_filtered = df.copy()
    for col in columns:
        signal = df[col]
        signal_pad = np.pad(signal, (kernel_size - 1, 0), 'edge')
        kernel = np.ones(kernel_size)
        signal_conv = np.convolve(signal_pad, kernel, mode='valid') / kernel_size
        df_filtered[col] = signal_conv

    df_corrected = df_filtered.shift(-offset) # correct filter delay
    df_corrected = df_corrected.iloc[:-offset] # delete last rows

    return df_corrected