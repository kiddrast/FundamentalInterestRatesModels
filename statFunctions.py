import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import scipy.stats as stats

def jb_test(data: np.array) -> df:

    '''

    Gives a df containing stat test e p-value for each path

    '''

    steps, paths = data.shape
    jb_summary = np.zeros((2, paths))

    for i in range(0,paths):
        col = data[:,i]
        mu = np.mean(col)
        std = np.std(col)             
        z = (col - mu) / std
        skewness = np.mean(z**3)
        kurtosis = np.mean(z**4)
        jb_stat  = (steps/6) * (skewness**2 + ((kurtosis - 3)**2)/4)
        p_value = 1.0 - stats.chi2.cdf(jb_stat, df=2)

        jb_summary[0,i] = jb_stat
        jb_summary[1,i] = p_value

    return df(jb_summary).rename(index={0:'jb stat', 1:'p value'})



def auto_correlation_function(data: np.array, p: int) -> df:

    '''

    Computes the acf up to lag p for every path. 

    '''

    _, paths = data.shape
    acf_summary = np.zeros((p+1, paths))

    for i in range(0,paths):

        col = data[:,i]
        mu = np.mean(col)
        var = np.sum((col - mu)**2) 
        acf_col = np.zeros(p + 1)
        acf_col[0] = 1

        for k in range(1, p + 1):
            cov = np.sum((col[k:] - mu) * (col[:-k] - mu))
            acf_col[k] = cov / var
        
        acf_summary[:,i] = acf_col.T

    return df(acf_summary)



def plot_acf(acf_summary: df, steps: int) -> None:

    '''
    
    Plots the acf function
    
    '''

    data = acf_summary.to_numpy()
    paths = data.shape[1]
    lags = np.arange(acf_summary.shape[0])
    conf = 1.0 / np.sqrt(steps)

    plt.figure(figsize=(8, 5))

    # Plot ACF for each path
    for i in range(0,paths):
        plt.plot(lags, data[:, i], alpha=0.7)

    # Confidence intervals around 0
    plt.axhline( conf, color='red', linewidth=1)
    plt.axhline(-conf, color='red', linewidth=1)

    # Plot 
    plt.axhline(0, color='black', linewidth=1)
    plt.title("Autocorrelation Function (ACF)")
    plt.xlabel("Lag")
    plt.ylabel("ACF Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return



def compute_moments(data: np.array) -> df:

    '''
    
    Gives a df with the first fourth moments
    
    '''

    paths = data.shape[1]
    moments = np.zeros((4, paths))

    for i in range(0,paths):
        col = data[:,i]
        mu = np.mean(col)
        var = np.var(col)
        std = np.std(col)             
        z = (col - mu) / std
        skewness = np.mean(z**3)
        kurtosis = np.mean(z**4)

        moments[:,i] = np.array([mu, var, skewness, kurtosis]).T

    return df(moments).rename(index={0: 'mean', 1: 'variance', 2: 'skewness', 3: 'kurtosis'})



def qq_plot(data: np.array, dist='normal', ncols=3, df=8) -> None:

    _, paths = data.shape
    nrows = int(np.ceil(paths / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.flatten()

    for i in range(paths):
        if dist == 'normal':
            stats.probplot(data[:, i], dist="norm", plot=axes[i])
        elif dist == 't':
            stats.probplot(data[:, i], dist="t", sparams=(df,), plot=axes[i])

        axes[i].set_title(f"Q-Q Plot Path {i}")
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

    return
















### For testing and debugging
if __name__ == "__main__":

    import autoregressive_oop as ar

    model = ar.AutoRegressive(steps=1_000, paths=6, a=np.array([0.2, 0.3, 0.2]), start=0)
    data = model.generate()
