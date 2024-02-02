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
    """
    Convert dataframe time columns to datetime format.

    This function iterates over all columns of the provided dataframe. If any column
    contains string representations of datetime in the format 'YYYY-MM-DD HH:MM:SS',
    it converts that column to a datetime datatype and renames it to "datetime".

    Parameters:
    - df (pandas.DataFrame): The input dataframe with one or more columns containing 
                            string representations of datetime values.

    Returns:
    - pandas.DataFrame: A dataframe with the relevant column converted to datetime format
                        and renamed to "datetime".
    """

    for x in df.columns:
        if df[x].astype(str).str.match(r'\d{4}-\d{2}-\d{2} \d{2}\:\d{2}\:\d{2}').all():
            df = df.rename(columns={x: "datetime"})
            df["datetime"] = pd.to_datetime(df['datetime'])
    return df

def convert_df_to_df_with_datetime_index(df):
    """
    Convert dataframe to have a datetime index.

    This function checks if the index of the provided dataframe is of datetime type.
    If not, it sets the 'datetime' column of the dataframe as the index.

    Parameters:
    - df (pandas.DataFrame): The input dataframe, which should contain a 'datetime'
                            column if the index is not already a datetime type.

    Returns:
    - pandas.DataFrame: A dataframe with its index set to a datetime type, either 
                        by using the existing datetime index or by setting the 'datetime' 
                        column as the index.
    """

    if df.index.inferred_type == 'datetime64':
        return df
    else:
        return df.set_index('datetime')

def make_df_continuous_in_time(df, max_minutes=30):
    """
    Make the dataframe continuous in time, filling in missing time intervals.

    This function takes a dataframe with a datetime index and ensures continuity in time.
    It identifies gaps larger than a specified maximum number of minutes, divides the dataframe 
    into continuous segments, and then resamples these segments to a one-minute frequency, 
    filling missing values with mean interpolation.

    Parameters:
    - df (pandas.DataFrame): The input dataframe with a datetime index.
    - max_minutes (int, optional): The maximum allowed gap in minutes before a new 
                                segment is considered. Default is 30 minutes.

    Returns:
    - list: A list of pandas.DataFrame objects, each representing a continuous time 
            segment of the original dataframe, resampled to a one-minute frequency.
    """

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
    """
    Convert multiple dataframes to the same frequency by skipping rows.

    This function iterates through a list of dataframes and, for each dataframe, finds the 
    starting row (up to a given number of initial rows to skip) that minimizes the total number 
    of NaN values when skipping a fixed number of rows. It then creates new dataframes with 
    reduced frequency by skipping the specified number of rows, starting from the identified 
    optimal row.

    Parameters:
    - dfs (list of pandas.DataFrame): List of dataframes to be converted to the same frequency.
    - rows_to_skip (int, optional): Number of rows to skip to reduce the frequency. Default is 2.

    Returns:
    - list: A list of pandas.DataFrame objects with reduced frequency, aligned by minimizing NaN values.
    """

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
    """
    Split dataframes based on consecutive NaN values in a specific column.

    This function takes a list of dataframes and splits each dataframe into multiple 
    sub-dataframes based on the number of consecutive NaN values in a specific column ('T_ret'). 
    If the number of consecutive NaNs exceeds a specified maximum, a new sub-dataframe is created. 
    Only sub-dataframes without exceeding the maximum consecutive NaNs are retained.

    Parameters:
    - dfs (list of pandas.DataFrame): List of dataframes to be split.
    - max_consecutive_nans (int): Maximum allowed consecutive NaN values to start a new split.

    Returns:
    - list: A list of pandas.DataFrame objects, each representing a segment of the original 
            dataframes without exceeding the specified maximum consecutive NaN values.
    """

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
    """
    Fill NaN values in dataframes and retain only those with length above a threshold.

    This function iterates over a list of dataframes, filling NaN values using linear interpolation. 
    It then filters out and retains only those dataframes whose length exceeds a specified threshold.

    Parameters:
    - dfs (list of pandas.DataFrame): List of dataframes to process.
    - thresh_len (int, optional): The minimum length required for a dataframe to be kept. Default is 200.

    Returns:
    - list: A list of pandas.DataFrame objects with NaN values filled and length exceeding the specified threshold.
    """

    dfs_valid = []
    for df in dfs:
        length=len(df)
        if length > thresh_len:
            dfs_valid.append(df.interpolate(method='linear'))
    
    return dfs_valid

def rearrange_and_keep_important_columns(dfs, columns):
    """
    Rearrange a list of dataframes to keep only specified columns.

    This function iterates over a list of dataframes, rearranging each dataframe to include 
    only the specified columns in the provided order.

    Parameters:
    - dfs (list of pandas.DataFrame): List of dataframes to be rearranged.
    - columns (list of str): List of column names to keep in the rearranged dataframes.

    Returns:
    - list: A list of pandas.DataFrame objects with columns rearranged and limited to those specified.
    """

    dfs_columns=[]
    for df in dfs:
        dfs_columns.append(df[columns])

    return dfs_columns

def first_preprocessing_step(df, max_minutes, rows_to_skip, max_consecutive_nans, thresh_len, columns):
    """
    Perform the first step of preprocessing on a dataframe.

    This function applies a series of preprocessing steps on a dataframe: converting time columns 
    to datetime, setting datetime as index, making the dataframe continuous in time, reducing 
    frequency, splitting based on consecutive NaNs, filling NaNs, keeping long dataframes only, 
    and rearranging to keep important columns. It combines previously defined functions for each step.

    Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - max_minutes (int): Maximum gap in minutes for continuous time segmentation.
    - rows_to_skip (int): Number of rows to skip for frequency reduction.
    - max_consecutive_nans (int): Maximum allowed consecutive NaN values for splitting.
    - thresh_len (int): Minimum length required for a dataframe to be kept.
    - columns (list of str): List of column names to keep after rearrangement.

    Returns:
    - list: A list of processed pandas.DataFrame objects.
    """

    df_date = convert_df_time_column_to_datetime(df)
    df_index = convert_df_to_df_with_datetime_index(df_date)
    dfs_continuous = make_df_continuous_in_time(df_index,  max_minutes)
    dfs_reduced = convert_dfs_variables_to_same_frequency(dfs_continuous, rows_to_skip)
    dfs_trim = split_dfs_based_on_consecutive_nans(dfs_reduced, max_consecutive_nans)
    dfs_valid = fill_dfs_nans_and_keep_long_dfs_only(dfs_trim, thresh_len)
    dfs_columns = rearrange_and_keep_important_columns(dfs_valid, columns)

    return dfs_columns

def remove_weekends(df):
    """
    Remove weekend days from a dataframe based on its datetime index.

    This function filters out weekend days from a dataframe with a datetime index. It identifies 
    weekdays and weekends, groups the dataframe by continuous weekday segments, and then returns 
    a list of dataframes corresponding to these weekday segments only, excluding weekends.

    Parameters:
    - df (pandas.DataFrame): The input dataframe with a datetime index.

    Returns:
    - list: A list of pandas.DataFrame objects, each representing a continuous segment of weekdays.
    """

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
    """
    Remove data corresponding to a specific day from the dataframe.

    This function checks if the specified day exists in the dataframe's index. If it does, 
    rows corresponding to that specific day are dropped. Otherwise, the original dataframe 
    is returned without any changes.

    Parameters:
    - df (pandas.DataFrame): The input dataframe with a datetime index.
    - removed_day (datetime-like, str, int, or Timestamp): The specific day to be removed from the dataframe.

    Returns:
    - pandas.DataFrame: The modified dataframe with the specified day removed, or the original 
                        dataframe if the specified day is not present.
    """

    # Check if removed_day exists in the DataFrame's index
    if removed_day in df.index:
        # Drop the rows corresponding to removed_day
        return df.drop(df.loc[removed_day].index)
    else:
        # If removed_day is not in the index, return the original DataFrame
        return df

def remove_non_working_hours(df, strating_hour='04:32', ending_hour='18:30'):
    """
    Remove non-working hours from a dataframe and retain days with the same length as the longest day.

    This function filters a dataframe to keep only data within specified working hours on weekdays. 
    It further processes the dataframe to ensure that only days with the same number of data points 
    as the longest day are retained.

    Parameters:
    - df (pandas.DataFrame): The input dataframe with a datetime index.
    - strating_hour (str, optional): The starting hour of the working period in 'HH:MM' format. Default is '04:32'.
    - ending_hour (str, optional): The ending hour of the working period in 'HH:MM' format. Default is '18:30'.

    Returns:
    - list: A list of pandas.DataFrame objects, each representing a day within the working hours and 
            having the same length as the longest day in the dataset.
    """

    df = df.between_time(strating_hour,ending_hour)
    df = df[df.index.dayofweek < 5]
    dfs_day_working_hours = [group[1] for group in df.groupby(df.index.date)]
    dfs_day_working_hours = [df for df in dfs_day_working_hours if len(df) == len(max(dfs_day_working_hours, key=len))]
    
    return dfs_day_working_hours    

def filter_signal_non_causal(df, columns, kernel_size=5):
    """
    Apply a non-causal filtering to specified columns of a dataframe.

    This function performs a non-causal filtering on the specified columns of the dataframe. 
    It applies a convolution operation with a uniform kernel of a specified size, effectively 
    smoothing the signal in the specified columns. The operation is non-causal as it uses future 
    data points for filtering.

    Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - columns (list of str): List of column names in the dataframe to be filtered.
    - kernel_size (int, optional): The size of the convolution kernel. Default is 5.

    Returns:
    - pandas.DataFrame: The dataframe with the specified columns filtered.
    """
        
    df_filtered = df.copy()
    outside = kernel_size//2
    for col in columns:
        signal = df[col]
        signal_pad = np.pad(signal, (outside, outside), 'edge')
        signal_conv = np.convolve(signal_pad, np.ones((kernel_size,))/kernel_size, mode='valid')
        df_filtered[col] = signal_conv

    return df_filtered

def filter_signal_causal(df, columns, kernel_size=5):
    """
    Apply a causal filtering to specified columns of a dataframe.

    This function performs a causal filtering on the specified columns of the dataframe. 
    It applies a convolution operation with a uniform kernel, ensuring that only past 
    data points are used for filtering, making the operation causal. After filtering, 
    the resulting signal is adjusted to correct for the filter delay.

    Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - columns (list of str): List of column names in the dataframe to be filtered.
    - kernel_size (int, optional): The size of the convolution kernel. Default is 5.

    Returns:
    - pandas.DataFrame: The dataframe with the specified columns causally filtered and adjusted for filter delay.
    """
    
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

def second_preprocessing_step(df, columns, columns_to_filter, strating_hour, ending_hour, removed_day, kernel_size=5, projection = 30, ):
    """
    Perform the second step of preprocessing on a dataframe.

    This function applies several preprocessing steps including non-causal filtering, creating time-related 
    features, removing specific days and non-working hours, and preparing data for prediction. It combines 
    earlier defined functions and additional operations to transform the data into a format suitable for 
    predictive modeling. 

    Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - columns (list of str): List of column names to include in the final dataframe.
    - columns_to_filter (list of str): Columns to be non-causally filtered.
    - strating_hour (str): The starting hour of the working period in 'HH:MM' format.
    - ending_hour (str): The ending hour of the working period in 'HH:MM' format.
    - removed_day (datetime-like, str, int, or Timestamp): Specific day to be removed from the dataframe.
    - kernel_size (int, optional): The size of the kernel for non-causal filtering. Default is 5.
    - projection (int, optional): Number of rows to shift for future value prediction. Default is 30.

    Returns:
    - pandas.DataFrame: The preprocessed dataframe ready for predictive modeling.
    """

    df = filter_signal_non_causal(df, columns_to_filter, kernel_size)
    df['Diff_temp'] = df['T_imp'] - df['T_ret']
    df['Day_week'] = df.index.to_series().dt.dayofweek
    df['Hours'] = df.index.to_series().dt.hour
    df['T_ret_in_1h'] = df['T_ret'].shift(-projection) # 1 hour is 30 rows, since between rows there is a 2 min interval
    df = df.iloc[:-projection]
    df['Hours_sin'] = np.sin(2 * np.pi *   df['Hours']/24.0)
    df['Hours_cos'] = np.cos(2 * np.pi *   df['Hours']/24.0)
    df['Day_week_sin'] = np.sin(2 * np.pi * df['Day_week']/7)
    df['Day_week_cos'] = np.cos(2 * np.pi * df['Day_week']/7)
    df = df.drop(['Day_week', 'Hours'], axis=1)
    df = remove_specific_day(df, removed_day)
    dfs_day_working_hours = remove_non_working_hours(df, strating_hour, ending_hour)
    dfs_for_prediction = []

    for i,df_day in enumerate(dfs_day_working_hours):
        df_day=df_day.reset_index().drop('datetime', axis=1)
        df_day['Day'] = i
        dfs_for_prediction.append(df_day)

    # Check if dfs_for_prediction is empty
    if dfs_for_prediction:
        df_for_prediction = pd.concat(dfs_for_prediction, ignore_index=True)
    else:
        # Create an empty DataFrame with specified columns if dfs_for_prediction is empty
        df_for_prediction = pd.DataFrame(columns=columns)

    df_for_prediction = df_for_prediction[columns]

    return df_for_prediction