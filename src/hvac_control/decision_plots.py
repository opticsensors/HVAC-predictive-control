import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def compute_timeseries(df):
    """
    Prepare a dataframe for time series analysis.

    This function resets the index of the given dataframe and drops the 'datetime' column. 
    It is typically used to prepare the dataframe for time series analysis where 'datetime' 
    is not needed as a separate column.

    Parameters:
    - df (pandas.DataFrame): The input dataframe with 'datetime' as one of its columns.

    Returns:
    - pandas.DataFrame: The dataframe with its index reset and 'datetime' column dropped.
    """

    return df.reset_index().drop('datetime', axis=1)

def timeseries_plot(df, columns, time, grid_size=(3,4), main_title='Time series plot', plot_size=(500,800), margin=300, spacing =435, dpi=200.):
    """
    Create a time series plot for specified columns of a dataframe.

    This function generates a grid of time series plots for specified columns of a dataframe. 
    Each subplot displays the time series of one column. The function allows customization of 
    various aspects like grid size, plot size, margins, spacing, and dpi for high-quality plots. 
    Additionally, it computes time-related columns ('minutes', 'hours', 'days') for x-axis representation 
    and returns the plot as an image in OpenCV format.

    Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - columns (list of str): List of column names to be plotted.
    - time (str): The column name representing time on the x-axis.
    - grid_size (tuple, optional): Size of the grid for subplots (rows, cols). Default is (3, 4).
    - main_title (str, optional): Main title of the plot. Default is 'Time series plot'.
    - plot_size (tuple, optional): Size of the plot (height, width) in pixels. Default is (500, 800).
    - margin (int, optional): Margin size in pixels. Default is 300.
    - spacing (int, optional): Spacing between plots in pixels. Default is 435.
    - dpi (float, optional): Dots per inch for the plot. Default is 200.

    Returns:
    - numpy.ndarray: An image of the generated plot in OpenCV format.
    """

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
    df['minutes'] = df.index * 2
    df['hours'] = df['minutes'] / 60
    df['days'] = df['hours'] / 24
    for ax, col, title in zip(axes.flatten(), columns, columns):
        ax.plot(df[time], df[col])
        ax.title.set_text(title)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlabel(time)
        ax.set_ylabel('value') 

    # save figure to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf)
    plt.close('all')
    return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

def compute_correlations(df, x_columns, y_column, max_shift, circular_shift=True):
    """
    Compute correlations between specified columns and a reference column with varying shifts.

    This function calculates the Pearson correlation coefficients between a set of specified 
    columns (x_columns) and a reference column (y_column) over a range of shifts up to a 
    maximum specified shift. It supports both circular and non-circular shifting of the 
    x_columns to compute these correlations. The result is a dataframe where each row 
    represents the correlations for a specific shift value.

    Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - x_columns (list of str): List of column names to be correlated with the reference column.
    - y_column (list of str): The reference column name.
    - max_shift (int): The maximum number of shifts for which to compute correlations.
    - circular_shift (bool, optional): Flag to perform circular shifting. Default is True.

    Returns:
    - pandas.DataFrame: A dataframe containing correlation coefficients for each shift and column.
    """

    list_of_dict = []
    df_sorted = df[x_columns+y_column]

    for shift in range(max_shift):
        df_copy=df_sorted.copy()
        
        if circular_shift:
            df_shifted=pd.concat([df_copy[x_columns].iloc[-shift:], df_copy[x_columns].iloc[:-shift]])
            df_copy[x_columns]=df_shifted.to_numpy()
        else:
            df_copy[x_columns] = df_copy[x_columns].shift(shift)
            df_copy = df_copy.iloc[shift:]

        correaltions = df_copy.corr(method='pearson').iloc[:,-1] # TODO make it work so we dont have to assume the reference signal is the rightmost column
        d_correlations = correaltions.to_dict()
        list_of_dict.append(d_correlations)

    df_corr = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))    
    df_to_plot = df_corr[x_columns].copy()

    return df_to_plot

def correlation_plot(df, x_columns, main_title='Correlation plot', x_ticks=8, plot_size=(600,1000), margin=600, spacing =925, dpi=200.):
    """
    Create a plot visualizing correlations with varying shifts.

    This function generates a plot to visualize the computed correlation coefficients between 
    a set of specified columns and a reference column over varying shifts. It allows 
    customization of the plot size, margins, spacing, and dpi for high-quality visualization. 
    Additionally, it sets up a secondary x-axis to indicate the sample number corresponding 
    to the shift values. The plot is returned as an image in OpenCV format.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing correlation data.
    - x_columns (list of str): List of column names whose correlations are to be plotted.
    - main_title (str, optional): Main title of the plot. Default is 'Correlation plot'.
    - x_ticks (int, optional): Number of ticks on the x-axis. Default is 8.
    - plot_size (tuple, optional): Size of the plot (height, width) in pixels. Default is (600, 1000).
    - margin (int, optional): Margin size in pixels. Default is 600.
    - spacing (int, optional): Spacing between plots in pixels. Default is 925.
    - dpi (float, optional): Dots per inch for the plot. Default is 200.

    Returns:
    - numpy.ndarray: An image of the generated plot in OpenCV format.
    """
        
    max_h, max_w = plot_size
    width = (max_w+margin+spacing)/dpi # inches
    height= (max_h+margin+spacing)/dpi

    left = margin/dpi/width #axes ratio
    bottom = margin/dpi/height
    wspace = spacing/float(max_w)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    # Add some extra space for the second axis at the bottom
    fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom, 
                        wspace=wspace, hspace=wspace)
    fig.suptitle(main_title, size=20)
    max_shift = len(df)
    df['minutes'] = df.index * 2
    df['hours'] = df['minutes'] / 60
    df.plot(x='hours', y=x_columns, ax=ax1)
    ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax1.set_xlabel('Past hours')
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

def compute_frequencies(df, columns, frate, max_freq):
    """
    Compute the frequency spectrum for specified columns of a dataframe.

    This function calculates the frequency spectrum using the Fast Fourier Transform (FFT) 
    for each specified column in the dataframe. It removes the zero frequency component, 
    focuses on positive frequencies, and limits the frequencies to a specified maximum 
    frequency. The result is a dataframe containing frequencies and their corresponding 
    magnitudes for each column.

    Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - columns (list of str): List of column names for which the frequency spectrum is computed.
    - frate (float): The sampling rate of the data (samples per second).
    - max_freq (float): The maximum frequency to include in the output.

    Returns:
    - pandas.DataFrame: A dataframe containing frequencies and magnitudes for each specified column.
    """
    
    dict_to_save={}
    n = df.shape[0]
    for col in columns:
        var_fft = np.fft.fft(df[col])
        var_fft[0] = 0
        var_mag = np.abs(var_fft)
        freqs = np.fft.fftfreq(n, 1./frate) # cycles/second
        positive_freqs = freqs[:n//2] 
        new_var_mag = var_mag[:n//2]
        limited_freqs = positive_freqs[positive_freqs<max_freq]
        limited_mag = new_var_mag[positive_freqs<max_freq]
        dict_to_save['freq_'+col] = limited_freqs
        dict_to_save['mag_'+col] = limited_mag

    df = pd.DataFrame(dict_to_save, columns=list(dict_to_save.keys()))   

    return df

def frequency_plot(df, columns, grid_size=(3,4), main_title='Frecuency plot', x_ticks=8, plot_size=(500,800), margin=300, spacing =500, dpi=200.):
    """
    Create a plot visualizing the frequency spectrum for specified columns.

    This function generates a grid of frequency spectrum plots for specified columns of a dataframe. 
    Each subplot displays the frequency spectrum for one column. The function allows customization 
    of various aspects like grid size, plot size, margins, spacing, and dpi for high-quality plots. 
    It also sets up a secondary x-axis to indicate the period in hours corresponding to the 
    frequency values. The plot is returned as an image in OpenCV format.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing frequency data.
    - columns (list of str): List of column names whose frequency spectrums are to be plotted.
    - grid_size (tuple, optional): Size of the grid for subplots (rows, cols). Default is (3, 4).
    - main_title (str, optional): Main title of the plot. Default is 'Frequency plot'.
    - x_ticks (int, optional): Number of ticks on the x-axis. Default is 8.
    - plot_size (tuple, optional): Size of the plot (height, width) in pixels. Default is (500, 800).
    - margin (int, optional): Margin size in pixels. Default is 300.
    - spacing (int, optional): Spacing between plots in pixels. Default is 500.
    - dpi (float, optional): Dots per inch for the plot. Default is 200.

    Returns:
    - numpy.ndarray: An image of the generated plot in OpenCV format.
    """

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

        ax2 = ax.twiny()

        # Add some extra space for the second axis at the bottom
        fig.subplots_adjust(bottom=0.2)

        ax.plot(df['freq_'+col], df['mag_'+col])
        ax.title.set_text(title)
        ax.set_xlabel('Freq Hz')
        ax.set_ylabel('Magnitude') 
        ax.margins(x=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(x_ticks))
        ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
        new_tick_locations = np.arange(x_ticks+1)
        max_freq=df['freq_'+col].max()
        
        def tick_function(X):
            
            X=X*(max_freq)/x_ticks
            with np.errstate(divide='ignore'):
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