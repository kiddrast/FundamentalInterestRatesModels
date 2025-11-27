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
        else:
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
    start_row = np.full(shape=data[0,:].shape, fill_value=start)
    for i in range(0,p):
        data[i,:] = start_row

    # Generate errors
    epsilon = generate_errors(data, dist, error_var, degree_f, wald_mean)

    # Get coefficients
    a_0 = a[0]
    a = a[1:][::-1].reshape(1, p)    #(1xp)
    
    # Fill data
    for i in trange(p, steps):
        data[i,:] = (a_0 + a @ data[i-p:i,:] + epsilon[i,:]).ravel() # (paths,) , before .ravel() the shape is (1xpaths)
        
    print(f'{paths} different AR({p}) processes of {steps - p + 2} steps have been generated with increments following {dist} distribution') 

    return data



def fit_ar_ols(data: np.ndarray, p: int) -> np.ndarray:  
    
    '''

    Returns an array of dimension (len(a) x paths) of coefficients computed through OLS up to lag p

    '''

    # Auxiliary function to fit a single path
    def fit_col(col, p):

        # Preparing X and Y
        Y = col[p:,:]        # (steps-p x 1)
        X_0 = np.ones_like(Y)  # (steps-p x 1)   
        cols = [X_0]
        
        for i in range(1,p+1):
            X_i = col[p-i:-i,:]
            cols.append(X_i)
            
        X = np.hstack(cols)  # Populating X with y_t-1, y_t-2, ... , y_t-p

        a_hat = np.linalg.inv(X.T @ X) @ (Y.T @ X).T

        return a_hat 

    # Iterate fit col to every path
    steps, paths = data.shape
    coefficients = np.zeros((p+1,paths)) 

    for i in range(0,paths):
        col = data[:,i].reshape(steps,1) # (steps x 1)
        a_hat = fit_col(col, p=p).ravel()
        coefficients[:,i] = a_hat
    
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
    
    After fitting an AR(p) this function returns an arrays of dimension steps x paths. By default returns std residuals.

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
        window = y_hat[i-1:i-p-1:-1, :]    # (p,paths)    
        y_hat[i, :] = a_0 + np.sum(a * window, axis=0)

    # Compute and return errors
    eta     = data - y_hat         

    if std_residuals:
        epsilon = eta / np.std(eta, axis=0, keepdims=True)  
        return epsilon  # std residuals            
    else:
        return eta     # residuals
    




if __name__ == '__main__':
    data = generate_ar(steps=100000, paths=3, a=np.array([0.2, 0.9, 0.1]), start=0, dist='normal')

    print(fit_ar_ols(data, p=2))