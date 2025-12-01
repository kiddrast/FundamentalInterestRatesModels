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



def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:

    '''
    
    Returns the rolling mean of the given array computed on window timespan
    
    '''

    if window <= 1:
        return np.full_like(arr, arr, dtype=float)
    s = pd.Series(arr)
    rm = s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    return rm



def check_autocovariance_stationarity(y, max_lag=10, plot=True, window=25, min_points_for_regression=10):

    '''

    Checks empirially if aucovariance depends on lag k only.

    - Computes rolling mean of c_t^(k) 
    - Regresses c_t^(k) vs time

    '''

    y = y - np.mean(y) # De-mean
    T = len(y)
    results = {}

    for k in range(1, max_lag+1):
        products = (y[k:] * y[:-k])   # c_t^(k) for t = k..T-1
        mean_prod = products.mean()
        std_prod  = products.std(ddof=1)
        n_obs = products.size

        # rolling mean (aligned to the same index as `products`)
        rm = rolling_mean(products, window)

        # regression products ~ t
        regression = {'slope': np.nan, 'intercept': np.nan, 'pvalue': np.nan, 'rvalue': np.nan, 'r_squared': np.nan}
        if n_obs >= min_points_for_regression:
            t_idx = np.arange(k, T)  # time indices aligned with products
            # perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(t_idx, products)
            regression.update({
                'slope': float(slope),
                'intercept': float(intercept),
                'pvalue': float(p_value),
                'rvalue': float(r_value),
                'r_squared': float(r_value**2),
                'std_err': float(std_err),
                'n_obs': int(n_obs)
            })

        results[k] = {
            'products': products,
            'index': np.arange(k, T),   # corresponding t values
            'mean': float(mean_prod),
            'std': float(std_prod),
            'rolling_mean': rm,
            'regression': regression
        }

        if plot:
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(results[k]['index'], products, marker='.', linestyle='-', alpha=0.45, label='c_t^(k) (products)')
            ax.plot(results[k]['index'], rm, linestyle='-', linewidth=2, label=f'rolling mean (w={window})')
            # regression line (if computed)
            if not np.isnan(regression['slope']):
                t_idx = results[k]['index']
                reg_line = regression['intercept'] + regression['slope'] * t_idx
                ax.plot(t_idx, reg_line, linestyle='--', linewidth=1.5,
                        label=f"reg line: slope={regression['slope']:.3e}, p={regression['pvalue']:.3f}, R²={regression['r_squared']:.3f}")
            ax.hlines(mean_prod, results[k]['index'][0], results[k]['index'][-1], colors='r', linestyles='--', label=f'mean={mean_prod:.4f}')
            ax.set_title(f'Products c_t^(k) for k={k} (mean={mean_prod:.4f}, std={std_prod:.4f})')
            ax.set_xlabel('t'); ax.set_ylabel('c_t^(k)')
            ax.legend(loc='best'); ax.grid(True)
            plt.show()

    return #results












### For testing and debugging
if __name__ == "__main__":

    import ar_helpers.autoregressive_oop as ar

    model = ar.AutoRegressive(steps=1_000, paths=6, a=np.array([0.2, 0.3, 0.2]), start=0)
    data = model.generate()
