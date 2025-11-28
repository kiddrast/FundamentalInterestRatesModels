import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt # TODO: maybe plotly looks better
from tqdm import trange
import scipy.stats as stats
from IPython.display import display

from likelihoodFunctions import neg_loglik_normal_ar, neg_loglik_t_ar, neg_loglik_wald_ar
from scipy.optimize import minimize 



def generate_errors(data: np.ndarray, dist: str, error_var: float, degree_f: float, seed=42) -> np.ndarray:
    random_generator = np.random.default_rng(seed=seed)
    if dist == 'normal':
        epsilon = random_generator.normal(loc=0, scale=np.sqrt(error_var), size=data.shape)
    elif dist == 't':
        epsilon = (random_generator.standard_t(degree_f, size=data.shape)) * np.sqrt(error_var)
    return epsilon



def generate_ar(steps: int, paths: int, a=np.ndarray, start=0, dist='normal', error_var=1, degree_f=None, disable_progress=False) -> np.ndarray:

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
    epsilon = generate_errors(data, dist, error_var, degree_f)

    # Get coefficients
    a_0 = a[0]
    a = a[1:][::-1].reshape(1, p)    #(1xp)
    
    # Fill data
    for i in trange(p, steps, disable=disable_progress):
        data[i,:] = (a_0 + a @ data[i-p:i,:] + epsilon[i,:]).ravel() # (paths,) , before .ravel() the shape is (1xpaths)
    
    if not disable_progress:
        print(f'{paths} different AR({p}) processes of {steps - p + 2} steps have been generated with increments following {dist} distribution') 

    return data



def iterate_simulations(steps_list: list, paths: int,  a: np.ndarray, dist='normal', error_var=1, degree_f=None) -> dict:

    '''
    
    Returns a dictionary with elements of steps_list as keys and and np.ndarray of size (steps x paths)
    autoregressive processes. In other words iterates generate_ar() to many #steps

    '''

    simulations = {}

    for steps in steps_list:
        sim = generate_ar(steps=steps, paths=paths, a=a, dist=dist, error_var=error_var, degree_f=degree_f, disable_progress=True)
        simulations[steps] = sim
    
    return simulations



def iterate_fit_ols(simulations: dict, p: int, return_df=True) -> df | dict:

    '''
    
    By default returns a pandas dataframe with the average coefficients fitted for every # steps (averaged over paths)
    If return_df=False, it returns a dictionary containing the coefficients found for every # of steps, not averaged per path.
    
    '''

    coef = {}

    for steps in simulations:
        coef[steps] = fit_ar_ols(data=simulations[steps], p=p)

    if return_df:
        return df({k: np.mean(v, axis=1).ravel() for k, v in coef.items()})
    else:
        return coef



def characteristic_poly_roots(a: np.ndarray, show_plot=True) -> np.ndarray:

    '''

    Plots and returns the roots of the characteristic polynomial of an AR(p) process, a must be an array of coefficients: [a0, a1, a2, ..., ap]

    '''

    poly = np.concatenate([-a[:0:-1], np.array([1])])
    roots = np.roots(poly)

    if show_plot:
        # Scatter plot 
        plt.scatter(roots.real, roots.imag, marker='x')
        # Unitary circle
        circle = plt.Circle((0,0), 1, fill=False, color='gray')
        plt.gca().add_patch(circle)
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.gca().set_aspect('equal')
        plt.title("Characteristic Polynomial Roots")
        plt.show()
    
    return roots



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



def fit_ar_ML(y_t: np.ndarray, p: int, dist='normal', method='L-BFGS-B'):

    '''
    
    Returns a minimize.res object relative to the optimization (minimum) of the negative log-likelihood
    corresponding to 'dist' parameter.
    
    '''

    init_a0 = np.mean(y_t)
    init_coeff = np.zeros(shape=(p))
    init_sigma_2 = np.var(y_t)

    if dist == 'normal':

        x0 = np.concatenate([[init_a0], init_coeff, [init_sigma_2]])
        bounds = [(None,None)] * (p+1) + [(0.01,None)] # No bounds for coeff + Bounds ( >0) for variance
        res = minimize(fun=neg_loglik_normal_ar, x0=x0, args=(y_t,), method=method, bounds=bounds)
    
    elif dist == 't':

        init_nu = 6.5
        x0 = np.concatenate([[init_a0], init_coeff, [init_sigma_2], [init_nu]])
        bounds = [(None,None)] * (p+1) + [(0.01,None)] + [(0.01,None)] # No bounds for coeff + Bounds ( >0) for variance and nu
        res = minimize(fun=neg_loglik_t_ar, x0=x0, args=(y_t,), method=method, bounds=bounds)
    
    return res



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
        a = coefficients[1:].reshape(p, 1)[::-1]    
    else:
        a_0 = coefficients[0, :]                   # (paths,)
        a = coefficients[1:, :][::-1,:]            # (p,paths)
    
    # Generate data
    for i in trange(p, steps): 
        y_hat[i, :] = a_0 + np.sum(a * y_hat[i-p:i,:], axis=0)

    # Compute and return errors
    eta = data - y_hat         

    if std_residuals:
        epsilon = eta / np.std(eta, axis=0, keepdims=True)  
        return epsilon  # std residuals            
    else:
        return eta     # residuals
    


def plot_paths(data=None, size=(11,3),  title='AR processes'):
    plt.figure(figsize=size)
    plt.plot(data)
    plt.title(title)
    plt.grid(True)
    plt.show()








### test and debug

if __name__ == '__main__':
    data = generate_ar(steps=100000, paths=1, a=np.array([0.1, -0.5, 0.3, 0.4, -0.2]), start=0, dist='t', degree_f=7)

    print(fit_ar_ML(y_t=data, p=4, dist='t'))