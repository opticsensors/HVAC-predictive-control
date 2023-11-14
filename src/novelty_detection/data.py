import os
import pandas as pd
from novelty_detection import parameters

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

def load_data(name, data_type=None, header_names=None, separator=','):
    file_path = _file_path(name, data_type)
    if header_names is None: # data has column titles in first row 
        data = pd.read_csv(file_path, sep=separator)
    else:
        data = pd.read_csv(file_path, sep=separator, names=header_names)
    return data

def save_data(df, name, data_type='processed'):
    file_path=os.path.join(parameters.DATA_PATH, data_type, name)
    df.to_csv(file_path, sep=',')


