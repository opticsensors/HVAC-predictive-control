import numpy as np
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