import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functions.likelihoodFunctions import neg_loglik_var



def generate_sigma(k, bounds):

    rng = np.random.default_rng(seed=42)

    B = rng.uniform(bounds[0], bounds[1], (k, k))
    sigma = B @ B.T + np.diag(np.ones(k) * k * bounds[0]) 

    return sigma



def generate_errors(T, k, sigma) -> np.ndarray:

    rng = np.random.default_rng(seed=42)

    Z = rng.standard_normal((T, k))       # (T,k) All independent
    L = np.linalg.cholesky(sigma)         # Gives the lower triangular

    return Z @ L.T 



def generate_random_A(k, p, diag_dominant=False, offdiag_scale=0.5):

    rng = np.random.default_rng(seed=42)
    A = rng.standard_normal((p, k, k)) 

    # Scaling off diagonal elements
    if diag_dominant: # If diag dominant -> more autoregressive then cross regressive effect
        for i in range (p):
            diag = np.diag(np.diag(A[i]))
            off = A[i] - diag
            A[i] = diag + off * offdiag_scale

    # Scaling whole matrix
    return A



def companion_matrix(A):

    p, k, _ = A.shape
    M = np.zeros((k*p, k*p))

    # First block row: A1 A2 ... Ap-1 Ap
    for i in range(p):
        M[0:k, i*k:(i+1)*k] = A[i]

    # Identity blocks
    for block in range(1, p):
        M[block*k:(block+1)*k, (block-1)*k:block*k] = np.eye(k)
    return M



def spectral_radius(M):

    eigs = np.linalg.eigvals(M)
    rho = np.max(np.abs(eigs))  # rho is the max eigenvalue in abs value

    return rho, eigs



def scale_A_to_target(A, target=0.95, max_iter=5):

    for it in range(max_iter):
        M = companion_matrix(A)
        rho, eigs = spectral_radius(M)
        if rho <= target:
            return A, rho, eigs
        
        s = target / rho
        A = A * s

    # Final check
    M = companion_matrix(A)
    rho, eigs = spectral_radius(M)
    return A, rho, eigs



def generate_A_stationary(k, p, plot_eigs=True, diag_dominant=False, offdiag_scale=0.5):

    A = generate_random_A(k, p, diag_dominant, offdiag_scale)
    A, rho, eigs = scale_A_to_target(A)

    if plot_eigs:
        # Scatter plot 
        plt.scatter(eigs.real, eigs.imag, marker='x')
        # Unitary circle
        circle = plt.Circle((0,0), 1, fill=False, color='gray')
        plt.gca().add_patch(circle)
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.gca().set_aspect('equal')
        plt.title("Companion matrix eigenvalues")
        plt.show()
    return A



def generate_var(T: int, k: int, p: int, sigma: np.ndarray, A: np.ndarray, A_0= None, disable_progress=False) -> np.ndarray:

    if A_0 is None:
        A_0 = np.zeros(shape=(k,))
    
    epsilon = generate_errors(T, k, sigma)
    data = np.zeros(shape=(T,k))
    A = np.hstack(A)

    for i in trange(p, T, disable=disable_progress):

        window = data[i-p:i,:][::-1]
        vec = window.ravel(order='C') # Flatten for rows 
        
        data[i] = A_0 + (A @ vec).T + epsilon[i]

    return data



def iterate_simulations(steps_list: list, k: int, p: int,  A: np.ndarray, sigma: np.ndarray, A_0=None) -> dict:

    simulations = {}

    for steps in steps_list:
        sim = generate_var(T=steps, k=k, p=p, sigma=sigma, A=A, A_0=A_0, disable_progress=True)
        simulations[steps] = sim
    
    return simulations



def fit_var_OLS(data, p) -> np.ndarray:

    T, k = data.shape

    Y = data[p:,:]
    X = np.ones(shape=(T-p, 1+k*p))

    for i in range(0,p):
        X[:, 1+k*i : 1+k*(i+1)] = data[p-i-1 : T-i-1, :]

    A_hat = np.linalg.inv(X.T @ X) @ (X.T @ Y)

    A0_hat = A_hat[1,:]
    A_hat = A_hat[1:].reshape(p,k,k).transpose(0, 2, 1)

    return A0_hat , A_hat # shapes: (k,) , (p,k,k)



def fit_var_ML(Y, p):

    T, k = Y.shape

    # initialize
    A0 = np.zeros((1 + k*p, k))
    Sigma0 = np.cov(Y.T)
    L0 = np.linalg.cholesky(Sigma0)
    tril_idx = np.tril_indices(k) # Gives the indeces of lower triangular part
    Sigma_params0 = L0[tril_idx]
    
    x0 = np.concatenate([A0.ravel(), Sigma_params0])

    res = minimize(neg_loglik_var, x0, args=(Y, p), method='L-BFGS-B')
    
    # build A and sigma
    n_coef = (1 + k*p) * k
    A_hat = res.x[:n_coef].reshape(1 + k*p, k)
    Sigma_params_hat = res.x[n_coef:]
    L = np.zeros((k, k))
    L[tril_idx] = Sigma_params_hat
    Sigma_hat = L @ L.T

    A0_hat = A_hat[1,:]
    A_hat = A_hat[1:].reshape(p,k,k).transpose(0, 2, 1)
    
    return A0_hat, A_hat, Sigma_hat





# Testing and debugging
if __name__ == '__main__':

    k = 3
    T = 100
    p=3

    A = generate_A_stationary(k, p, plot_eigs=True, diag_dominant=False, offdiag_scale=0.5)
    sigma = generate_sigma(k, bounds=(0.1, 1.0))
    data = generate_var(T, k, p, sigma, A)

    A0_hat_OLS, A_hat_OLS = fit_var_OLS(data, p)
    A0_hat_ML, A_hat_ML, Sigma_hat = fit_var_ML(data, p)

    b = 0

    
