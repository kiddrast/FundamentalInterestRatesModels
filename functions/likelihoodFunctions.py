import numpy as np
from scipy.special import gammaln



def conditional_mean(y_t: np.ndarray, a: np.ndarray) -> np.ndarray:

    '''
    
    Returns the conditional mean for an AR(p) process: mu_t = a0 + a1*y_{t-1} + ... + ap*y_{t-p}
    
    '''

    T = y_t.size
    p = a.size - 1
    mu_t = np.zeros_like(y_t) 
    mu_t[:p] = y_t[:p] # In this way the first p values of u_t are not null and are equal to the observed ones

    a_0 = a[0]
    a = a[1:][::-1].reshape(1,p) # (1xp)

    for i in range(p, T):                 
        mu_t[i] = (a_0 + a @ y_t[i-p:i]).ravel()  
        
    return mu_t



##### NORMAL #####

def loglik_normal(y_t: np.ndarray, mu, sigma_2) -> float:

    '''
    
    Log-Likelihood of Normal RV. y_t must be a vector (Tx1)
    
    '''

    T = y_t.size
    term_1 = - 0.5 * T * np.log(2 * np.pi)
    term_2 = - 0.5 * T * np.log(sigma_2)
    term_3 = - ((y_t - mu)**2) / (2 * sigma_2)

    return term_1 + term_2 + np.sum(term_3, axis=0) 


def neg_loglik_normal_ar(a: np.ndarray, y_t: np.ndarray) -> float:

    '''
    
    Negative Log-Likelihood for fitting an AR(p) process with normal innovations. 
    a must be an array of parameters such that: [a0, a1, ..., ap, sigma^2].
    y_t must be of shape (steps x 1)
    
    '''

    p = a.size - 2
    sigma_2 = a[-1]
    mu_t = conditional_mean(y_t, a[:-1])

    return -loglik_normal(y_t[p:], mu_t[p:], sigma_2)



def multi_col_neg_loglik_normal_ar(a: np.ndarray, y_t: np.ndarray) -> float:

    '''
    
    Returns the sum of the log likelihood computed for each column
    
    '''

    _, col = y_t.shape

    liks = np.zeros(shape=col)

    for i in range(0, col):
        liks[0] = neg_loglik_normal_ar(a, y_t[:,i])
    
    return np.sum(liks)



##### STUDENT T #####

def loglik_t(y_t: np.ndarray, mu, sigma_2, nu) -> float:

    '''
    
    Log-Likelihood of Student t RV. y_t must be a vector (Tx1)
    
    '''
    
    T = y_t.size
    term_1 =   T * (gammaln((nu+1)/2) - gammaln(nu/2))
    term_2 = - 0.5 * T * np.log(nu * np.pi) 
    term_3 = - 0.5 * T * np.log(sigma_2)
    term_4 = - ((nu + 1)/2) * np.log(1 + (((y_t - mu)**2) / (sigma_2 * nu)))

    return term_1 + term_2 + term_3 + np.sum(term_4, axis=0)


def neg_loglik_t_ar(a: np.ndarray, y_t: np.ndarray) -> float:

    '''
    
    Negative LogLikilihood for fitting an AR(p) process with student t innovations. 
    a is an array of parameters such that: [a0, a1, ... ap, sigma^2, nu].
    
    '''

    p = a.size - 3
    nu = a[-1]
    sigma_2 = a[-2]
    mu_t = conditional_mean(y_t, a[:-2])

    return -loglik_t(y_t[p:], mu_t[p:], sigma_2, nu)


def multi_col_neg_loglik_t_ar(a: np.ndarray, y_t: np.ndarray) -> float:

    '''
    
    Returns the sum of the log likelihood computed for each column
    
    '''

    _, col = y_t.shape

    liks = np.zeros(shape=col)

    for i in range(0, col):
        liks[0] = neg_loglik_t_ar(a, y_t[:,i])
    
    return np.sum(liks)



##### INVERSE GAUSSIAN #####

def loglik_wald(y_t, mu, lam) -> float:

    '''
    
    Log-Likelihood of Inverse Gaussian RV. y_t must be a vector (Tx1)
    
    '''

    term1 = 0.5 * np.log(lam) - 1.5 * np.log(y_t)
    term2 = - lam * (y_t - mu)**2 / (2 * mu**2 * y_t)

    pointwise_loglik = term1 + term2 - 0.5*np.log(2*np.pi)

    return np.sum(pointwise_loglik, axis=0) 


def neg_loglik_wald_ar(a: np.ndarray, y_t: np.ndarray) -> float:

    '''
    
    Negative LogLikelihood for fitting an AR(p) process with wald innovations. 
    a is an array of parameters such that: [a0, a1, ... ap, lambda].
    
    '''

    p = a.size - 2
    lam = a[-1]
    mu_t = conditional_mean(y_t, a[:-1])

    return -loglik_wald(y_t[p:], mu_t[p:], lam)













#####################################################################
################################ VAR ################################
#####################################################################




def conditional_mean_var(Y, A, p):

    '''

    Y: T x k
    A: (1 + k*p) x k
    Returns mu_t: T x k

    '''

    T, k = Y.shape
    mu_t = np.zeros_like(Y)
    mu_t[:p, :] = Y[:p, :]
    
    for t in range(p, T):
        x_t = np.hstack([1] + [Y[t - lag - 1, :] for lag in range(p)])
        mu_t[t, :] = x_t @ A
    return mu_t



def loglik_normal_var(Y, mu, Sigma):

    '''

    Y, mu: T x k
    Sigma: k x k (covariance)

    '''

    T, k = Y.shape
    diff = Y - mu
    sign, logdet = np.linalg.slogdet(Sigma)
    inv_Sigma = np.linalg.inv(Sigma)
    term = np.einsum('ti,ij,tj->', diff, inv_Sigma, diff)  # sum_t (y_t-mu_t)' Sigma^-1 (y_t-mu_t)

    return -0.5 * T * (k * np.log(2 * np.pi) + logdet) - 0.5 * term



def neg_loglik_var(params, Y, p):

    '''

    params: first (1 + k*p)*k for coefficients, then k*(k+1)/2 for Sigma

    '''

    T, k = Y.shape
    n_coef = (1 + k*p) * k
    A = params[:n_coef].reshape(1 + k*p, k)
    
    # let's build sigma from coefficients using lower-triangular parametrization
    Sigma_params = params[n_coef:]
    L = np.zeros((k, k))
    tril_idx = np.tril_indices(k)
    L[tril_idx] = Sigma_params
    Sigma = L @ L.T  # garantees PD
    
    mu_t = conditional_mean_var(Y, A, p)
    
    return - loglik_normal_var(Y[p:, :], mu_t[p:, :], Sigma)