import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

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

def timeseries_plot(df, columns, grid_size, main_title='Time series plot', plot_size=(600,600), margin=150, spacing =435, dpi=200.):

    rows, cols= grid_size
    max_h, max_w = plot_size
    width = (cols*max_w+cols*margin+spacing)/dpi # inches
    height= (rows*max_h+rows*margin+spacing)/dpi

    left = margin/dpi/width #axes ratio
    bottom = margin/dpi/height
    wspace = spacing/float(max_w)

    fig, axes  = plt.subplots(rows,cols, figsize=(width,height), dpi=dpi)
    fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom, 
                        wspace=wspace, hspace=wspace)
    fig.suptitle(main_title, size=20)
    for ax, col, title in zip(axes.flatten(), columns, columns):
        ax.plot(df.index, df[col])
        ax.title.set_text(title)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlabel('date')
        ax.set_ylabel('value') 

    # save figure to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf)
    plt.close('all')
    return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

def correlation_plot(df, columns, max_shift,main_title='Correlation plot', x_ticks=8, plot_size=(600,600), margin=150, spacing =435, dpi=200.):

    list_of_dict = []

    for shift in range(max_shift):
        df_copy=df.copy()
        df_copy[columns] = df_copy[columns].shift(shift)
        df_remove = df_copy.iloc[shift:]
        correaltions = df_remove.corr(method='pearson').iloc[:,-1]
        d_correlations = correaltions.to_dict()
        list_of_dict.append(d_correlations)

    df_corr = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))

    max_h, max_w = plot_size
    width = (max_w+margin+spacing)/dpi # inches
    height= (max_h+margin+spacing)/dpi

    left = margin/dpi/width #axes ratio
    bottom = margin/dpi/height
    wspace = spacing/float(max_w)

    # fig, ax  = plt.subplots(figsize=(width,height), dpi=dpi)
    # fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom, 
    #                     wspace=wspace, hspace=wspace)
    # df_corr[columns].plot(ax=ax)
    # ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    # ax.title.set_text('Correlation plot')
    # ax.set_xlabel('Sample number')
    # ax.set_ylabel('Pearson correlation coefficient') 

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    # Add some extra space for the second axis at the bottom
    fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom, 
                        wspace=wspace, hspace=wspace)
    fig.suptitle(main_title, size=20)
    df_to_plot = df_corr[columns].reset_index()
    df_to_plot['minutes'] = df_to_plot['index'] * 2
    df_to_plot['minutes_str'] = pd.to_datetime(df_to_plot['minutes'], unit='m').dt.strftime('%H:%M')
    df_to_plot.plot(x='minutes_str', y=columns,ax=ax1)
    ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax1.set_xlabel('Past time')
    ax1.set_ylabel('Pearson correlation coefficient') 
    ax1.margins(x=0)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(x_ticks))

    new_tick_locations = np.arange(x_ticks+1)

    def tick_function(X):
        X=X*max_shift/x_ticks
        return X.astype(np.int64)

    # Move twinned axis ticks and label from top to bottom
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")

    # Offset the twin axis below the host
    ax2.spines["bottom"].set_position(("axes", -0.25))

    # Turn on the frame for the twin axis, but then hide all 
    # but the bottom spine
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)

    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.spines["bottom"].set_visible(True)

    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    ax2.set_xlabel(f"Sample number")

    # save figure to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf)
    plt.close('all')
    return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

def frequency_plot(df, columns, grid_size, frate, max_freq, main_title='Frecuency plot', plot_size=(600,600), margin=150, spacing =435, dpi=200.):

    rows, cols= grid_size
    max_h, max_w = plot_size
    width = (cols*max_w+cols*margin+spacing)/dpi # inches
    height= (rows*max_h+rows*margin+spacing)/dpi

    left = margin/dpi/width #axes ratio
    bottom = margin/dpi/height
    wspace = spacing/float(max_w)

    fig, axes  = plt.subplots(rows,cols, figsize=(width,height), dpi=dpi)
    fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom, 
                        wspace=wspace, hspace=wspace)
    fig.suptitle(main_title, size=20)
    n = df.shape[0]
    for ax, col, title in zip(axes.flatten(), columns, columns):
        var_fft = np.fft.fft(df[col])
        var_fft[0] = 0
        var_mag = np.abs(var_fft)
        freqs = np.fft.fftfreq(n, 1./frate) # cycles/second

        #ax.plot(freqs[:n//2], var_mag[:n//2])
        #ax.title.set_text(title)
        #ax.set_xlim([0, max_freq])

        ax2 = ax.twiny()

        # Add some extra space for the second axis at the bottom
        fig.subplots_adjust(bottom=0.2)

        ax.plot(freqs[:n//2], var_mag[:n//2])
        ax.title.set_text(title)
        ax.set_xlabel('Freq Hz')
        ax.set_ylabel('Magnitude') 
        ax.margins(x=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(8))
        ax.set_xlim([0, max_freq])

        new_tick_locations = np.arange(8+1)

        def tick_function(X):
            
            X=X*(4*1e-5)/8
            X=(1/X) / 3600
            X_list=[f"{x:.1f}" if not np.isinf(x) else 'inf' for x in X]
            return np.array(X_list)

        # Move twinned axis ticks and label from top to bottom
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        # Offset the twin axis below the host
        ax2.spines["bottom"].set_position(("axes", -0.25))

        # Turn on the frame for the twin axis, but then hide all 
        # but the bottom spine
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)

        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.spines["bottom"].set_visible(True)

        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations))
        ax2.set_xlabel(f"Period hours")

    # save figure to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf)
    plt.close('all')
    return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)