import numpy as np
from sklearn.utils import check_array, check_random_state

def argmin_first_two_axes(A):
    min_idx = A.reshape(-1, A.shape[2]).argmin(0)
    return np.column_stack(np.unravel_index(min_idx, A[:,:,0].shape))

def choose_random_sample(X, random_state=0):
    X = check_array(X)
    random_state = check_random_state(random_state)
    i = random_state.randint(X.shape[0])
    return X[i] # same as X[i, :]

def choose_random_sample_numpy(X):
    X = check_array(X)
    i = np.random.randint(low=0, high=X.shape[0])
    return X[i] # same as X[i, :]

def choose_random_array(size,  random_state=0, uniform=True):
    random_state = check_random_state(random_state)
    if uniform:
        som = random_state.random_sample(size)
    else:
        som = random_state.randn(size[0], size[1], size[2])
    return som

def choose_random_array_numpy(size, uniform=True):
    if uniform:
        som = np.random.random_sample(size=size)
    else:
        som = np.random.randn(size[0], size[1], size[2])
    return som