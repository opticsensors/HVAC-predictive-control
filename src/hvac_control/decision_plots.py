import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def compute_timeseries(df):

    return df.reset_index().drop('datetime', axis=1)

def timeseries_plot(df, columns, time, grid_size=(3,4), main_title='Time series plot', plot_size=(500,800), margin=300, spacing =435, dpi=200.):

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