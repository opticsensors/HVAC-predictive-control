import os
import pandas as pd
import cv2
from novelty_detection import parameters
from novelty_detection.preprocessing import *

def _find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
        
def _file_path(name, data_type=None):
    if data_type is None:
        file_path = _find(name,parameters.REPO_PATH)
    else:
        file_path = os.path.join(parameters.DATA_PATH, data_type, name)
    return file_path

def load_data(name, data_type=None, header_names=None, separator=',', index=False):
    file_path = _file_path(name, data_type)
    if header_names is None: # data has column titles in first row 
        data = pd.read_csv(file_path, sep=separator)
    else:
        data = pd.read_csv(file_path, sep=separator, names=header_names)
    if index: # data has a datetime column and we want to convert it to new index
        data = convert_df_time_column_to_datetime(data)
        data = convert_df_to_df_with_datetime_index(data)
    return data

def save_data(df, name, data_type='processed', index=True):
    file_path=os.path.join(parameters.DATA_PATH, data_type, name)
    df.to_csv(file_path, sep=',', index=index)

def save_img(img, name, data_type='plots'):
    file_path=os.path.join(parameters.DATA_PATH, data_type, name)
    cv2.imwrite(file_path, img)


