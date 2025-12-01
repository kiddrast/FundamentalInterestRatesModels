import numpy as np
import pandas as pd
from pandas import DataFrame as df


'''

Reference: Lan ZHANG, Per A. MYKLAND, and Yacine AÏT-SAHALIA - Tale of Two Time Scales: Determining Integrated Volatility With Noisy High-Frequency Data 

'''



def realized_volatility(log_prices = None, log_returns = None) -> np.array: 

    if log_prices and log_returns is None:
        log_returns = np.diff(log_prices)

    return np.sum(log_returns ** 2, axis=0).reshape(1,-1)   # (1 x paths)



def sparse_volatility(n: int, log_prices = None, log_returns = None) -> np.array: 

    if log_prices:
        return realized_volatility(log_prices = log_prices[::n])
    
    if log_returns:
        return df(log_returns).groupby(np.arange(len(log_returns)) // n).sum() # TODO: improve this, infact it fails if the lenght is not divisible by n



def optimal_sparse_volatility() -> np.array:

    return ... # TODO



def sampling_averaging_volatility(K: int, log_prices = None, log_returns = None) -> np.array:

    if log_prices and log_returns is None:
        log_returns = np.diff(log_prices)

    data = df(log_returns)
    data['mask'] = data.index // K
    
    return data.groupby('mask').apply(lambda r: np.sum(r**2)).mean()



def two_scales_volatility(K: int, log_prices = None, log_returns = None) -> np.array:

    if log_prices and log_returns is None:
        log_returns = np.diff(log_prices)

    vol_all = realized_volatility(log_returns=log_returns)
    vol_avg = sampling_averaging_volatility(K, log_returns=log_returns)

    return vol_avg - vol_all





### For testing and debugging
if __name__ == "__main__":

    import ar_helpers.autoregressiveFunctions as ar

    data_ = ar.generate_ar(steps=1_000, paths=1, a=np.array([0.2, 1]), start=100)

    rv_all = realized_volatility(log_returns=data)
    tsrv = two_scales_volatility(log_returns=data)




