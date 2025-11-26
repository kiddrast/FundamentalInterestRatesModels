import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt # TODO: maybe plotly looks better
from tqdm import trange
import scipy.stats as stats
from IPython.display import display



def generate_errors(data: np.ndarray, dist: str, error_var: float, degree_f: float, wald_mean: float, seed=42) -> np.ndarray:
    random_generator = np.random.default_rng(seed=seed)
    if dist == 'normal':
        epsilon = random_generator.normal(loc=0, scale=np.sqrt(error_var), size=data.shape)
    elif dist == 't':
        epsilon = random_generator.standard_t(degree_f, size=data.shape)
    elif dist == 'wald':
        if wald_mean <=0:
            raise('The mean of a wald distribution must be greater than 0!')
        epsilon = random_generator.wald(error_var, error_var, size=data.shape)
    
    return epsilon



def generate_ar(steps: int, paths: int, a=np.ndarray, start=0, dist='normal', error_var=1, degree_f=None, wald_mean=None) -> np.ndarray:

    '''

    Returns an array of dimension (steps x paths) with columns representing different paths
    of the same AR(P) process. The array 'a' must contain the constant and the coefficients. 

    '''

    p = a.size - 1

    # Initialize and add first rows
    data = np.empty((steps,paths), dtype=float)
    start_row = np.full(shape=(1,paths), fill_value=start)
    for i in range(0,p):
        data[1,:] = start_row

    # Generate errors
    epsilon = generate_errors(data, dist, error_var, degree_f, wald_mean)
    
    # Fill data
    for i in trange(p, steps):
        window = data[i-p:i, :][::-1, :] 
        data[i,:] = a[0] + a[1:].T @ window + epsilon[i,:]
        
    print(f'{paths} different AR({p}) processes of {steps - p + 2} steps have been generated with increments following {dist} distribution') 

    return data



def fit_ar_ols(data: np.ndarray, p: int) -> np.ndarray:  
    
    '''

    Returns an array of dimension (len(a) x paths) of coefficients computed through OLS up to lag p

    '''

    # Auxiliary function to fit a single path
    def fit_col(col: np.array, p: int) -> np.array:

        Y = col[p:,:]
        X = np.ones_like(Y)       # Initialize X with same shape of Y and full of 1 (in order to get the constant)
        
        for i in range(1,p+1):
            v = col[p-i:-i,:]
            X = np.hstack((X,v))   # Populating X with y_t-1, y_t-2, ... , y_t-p

        a_hat = np.linalg.inv(X.T @ X) @ (Y.T @ X).T
        return np.vstack((a_hat[0], a_hat[:0:-1]))

    # Iterate fit col to every path
    coefficients = np.zeros((p+1,np.shape(data)[1]))
    i=0
    for col in data.T:
        a_hat = fit_col(col.reshape(-1,1), p=p).reshape(-1)
        coefficients[:,i] = a_hat
        i += 1

    return coefficients


'''
def fit_ar_ML(data: np.array, p: int, dist='') -> np.array:

    if dist == 'normal':
        likelihood = 
    if dist == 't':
        likelihood = 
    if dist == 'wald':
        likelihood = 
'''


def get_residuals(data: np.ndarray, coefficients: np.ndarray, p: int, std_residuals = True) -> np.ndarray:

    '''
    
    After fitting an AR(p) this function returns an arrays of dimension steps x paths. By default returns and std residuals.

    '''

    # Preparing y_hat
    steps, _ = data.shape
    y_hat = np.zeros_like(data)
    for i in range(0,p):
        y_hat[i,:] = data[i,:]  # First p rows filled with initial values

    # Get coefficients
    if coefficients.ndim == 1:
        a_0 = coefficients[0]
        a = coefficients[1:].reshape(p, 1)    
    else:
        a_0 = coefficients[0, :]           # (paths,)
        a = coefficients[1:, :]            # (p,paths)

    # Generate data
    for i in trange(p, steps):
        window = y_hat[i-p:i, :][::-1,:]     # (p,paths)    
        y_hat[i, :] = a_0 + np.sum(a * window, axis=0)

    # Compute and return errors
    eta     = data - y_hat         

    if std_residuals:
        epsilon = eta / np.std(eta, axis=0, keepdims=True)  
        return epsilon  # std residuals            
    else:
        return eta     # residuals