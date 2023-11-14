from sklearn.preprocessing import MinMaxScaler 

def minmax_scaler(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled