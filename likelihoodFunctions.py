import numpy as np
from scipy.special import gammaln



def conditional_mean(y_t: np.ndarray, a: np.ndarray) -> np.ndarray:

    '''
    
    Returns the conditional mean for an AR(p) process: mu_t = a0 + a1*y_{t-1} + ... + ap*y_{t-p}
    
    '''

    T = y_t.size
    p = a.size - 1
    mu_t = np.empty_like(y_t, dtype=float) 
    mu_t[:p] = y_t[:p] # In this way the first p values of u_t are not null and are equal to the observed ones

    for i in range(p, T):                 
        mu_t[i] = a[0] + a[1:] @ y_t[i-1:i-p-1:-1]  # a1*y_{i-1} + ... + ap*y_{i-p} 
        
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


def neg_loglik_normal_ar(y_t: np.ndarray, a: np.ndarray) -> float:

    '''
    
    Negative LogLikilihood for fitting an AR(p) process with normal innovations. 
    a is an array of parameters such that: [a0, a1, ... ap, sigma^2].
    
    '''

    p = a.size - 2
    sigma_2 = a[-1]
    a = a[:-1]
    mu_t = conditional_mean(y_t, a)

    return -loglik_normal(y_t[p:], mu_t[p:], sigma_2)



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

    return term_1 + term_2 + term_3 + term_4 + np.sum(term_4, axis=0)


def neg_loglik_t_ar(y_t: np.ndarray, a: np.ndarray) -> float:

    '''
    
    Negative LogLikilihood for fitting an AR(p) process with student t innovations. 
    a is an array of parameters such that: [a0, a1, ... ap, sigma^2, nu].
    
    '''

    p = a.size - 3
    nu = a[-1]
    sigma_2 = a[-2]
    a = a[:-2]
    mu_t = conditional_mean(y_t, a)

    return -loglik_t(y_t[p:], mu_t[p:], sigma_2, nu)



##### INVERSE GAUSSIAN #####

def loglik_wald(y_t, mu, lam) -> float:

    '''
    
    Log-Likelihood of Inverse Gaussian RV. y_t must be a vector (Tx1)
    
    '''

    term1 = 0.5 * np.log(lam) - 1.5 * np.log(y_t)
    term2 = - lam * (y_t - mu)**2 / (2 * mu**2 * y_t)

    pointwise_loglik = term1 + term2 - 0.5*np.log(2*np.pi)

    return np.sum(pointwise_loglik, axis=0) 


def neg_loglik_t_ar(y_t: np.ndarray, a: np.ndarray) -> float:

    '''
    
    Negative LogLikilihood for fitting an AR(p) process with wald innovations. 
    a is an array of parameters such that: [a0, a1, ... ap, lambda].
    
    '''

    p = a.size - 2
    lam = a[-1]
    a = a[:-1]
    mu_t = conditional_mean(y_t, a)

    return -loglik_wald(y_t[p:], mu_t[p:], lam)




