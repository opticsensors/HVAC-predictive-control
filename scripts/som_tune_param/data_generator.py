import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hvac_control.som import SOM
from hvac_control.som_metrics import SOMmetrics



def generate(size, dim, dispersion, method):
    if method=='uniform':
        data=np.random.rand(size, dim)*dispersion - dispersion/2
    elif method=='std':
        data=np.random.normal(0, dispersion, size=(size, dim))
    return data

l_size=[250, 500, 1000, 2000]
l_dispersion=[0.25, 0.5, 1, 1.5, 2, 3, 4]
l_som_sizes=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]

list_of_dict = []
dict_param={}

for size in l_size:
    print('size', size)
    for disp in l_dispersion:
        print('disp', disp)
        for som_size in l_som_sizes:
                data = generate(size, dim=3, dispersion=disp, method='uniform')
                std = np.std(data)
                iqr = np.percentile(data, 75) - np.percentile(data, 25)
                data_range = np.max(data) - np.min(data)
                centroid = np.mean(data, axis=0)
                distances = np.linalg.norm(data - centroid, axis=1)
                centroid_std = np.std(distances)

                model=SOM(som_grid_size=(som_size,som_size))
                model=model.fit(data)
                metrics=SOMmetrics(model.som)
                freq_map=metrics.find_bmu_counts(data, model.som)
                zeros = np.sum(freq_map == 0)
                percent_of_zeros = 100 * zeros / (model.som.shape[0]*model.som.shape[1])  

                dict_param['size']=size
                dict_param['disp']=size
                dict_param['std']=std
                dict_param['centroid_std']=centroid_std
                dict_param['som_size']=som_size
                dict_param['unused_som']=percent_of_zeros
                list_of_dict.append(dict_param.copy())


#for convenience we convert the list of dict to a dataframe
df = pd.DataFrame(list_of_dict, columns=list(list_of_dict[0].keys()))
df.to_csv(path_or_buf='./data_vs_som.csv', sep=',',index=False)