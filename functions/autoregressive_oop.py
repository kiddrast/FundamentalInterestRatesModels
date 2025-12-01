import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt # TODO: maybe plotly looks better
from tqdm import trange
from scipy.stats import chi2
from IPython.display import display

class AutoRegressive:

    def __init__(self, steps: int, paths: int, a=np.array, start=0, dist='normal', error_var=1, df=None):

        self.steps     = steps
        self.paths     = paths
        self.a         = a
        self.p         = a.size - 1
        self.start     = start      # TODO: add the option to give different starting points for each level
        self.start_row = np.full(shape=(1,paths), fill_value=self.start)
        self.dist      = dist
        self.error_var = error_var
        self.df        = df         # Degree of freedom of t



    def generate(self) -> np.array:

        '''

        Returns an array of dimension steps x paths with columns representing different paths
        of the same AR(P) process.

        '''
        
        # Initialize data and add first row
        data = np.zeros((self.steps,self.paths), dtype=float)
        for i in range(0,self.p):
            data[i,:] = self.start_row

        # Generate errors
        random_generator = np.random.default_rng()
        if self.dist == 'normal':
            epsilon = random_generator.normal(loc=0, scale=np.sqrt(self.error_var), size=data.shape)
        elif self.dist == 't':
            epsilon = random_generator.standard_t(self.df, size=data.shape)

        # Fill data
        for i in trange(self.p, self.steps):
            data[i,:] = self.a[0] + self.a[1:].T @ data[i-self.p:i,:] + epsilon[i,:]

        print(f'{self.paths} different AR({self.p}) processes of {self.steps - self.p + 2} steps have been generated with increments following {self.dist} distribution') 

        self.data: np.array = data
        return self.data



    def plot_paths(self, data=None, size=(11,3),  title=None):

        if title is None:
            title = f'AR({self.p}) processes'

        plt.figure(figsize=size)
        plt.plot(data)
        plt.title(title)
        plt.grid(True)
        plt.show()



    def fit_ar_ols(self, p=None, data=None) -> np.array:  
        
        '''

        If the method si ols it gives an array of dimension len(self.a) x paths of coefficients
        By default data is the one generated through specific function, however if the function fit_ar() has been provided by other data, it is possible to pass it.
        (same for p)

        '''
        
        if p is None:
            p = self.p
        if data is None:
            data = self.data
        
        # Auxiliary function to fit a single path
        def fit_col(col: np.array, p: int) -> np.array:

            Y = col[p:,:]
            X = np.ones_like(Y)      # Initialize X with same shape of Y and full of 1 (in order to get the constant)
            
            for i in range(1,p+1):
                v = col[p-i:-i,:]
                X = np.hstack((X,v)) # Populating X with y_t-1, y_t-2, ... , y_t-p

            a_hat = np.linalg.inv(X.T @ X) @ (Y.T @ X).T
            return np.vstack((a_hat[0], a_hat[:0:-1]))

        # Iterate fit col to every path
        coefficients = np.zeros((self.p+1,np.shape(data)[1]))
        i=0
        for col in data.T:
            a_hat = fit_col(col.reshape(-1,1), p = self.p).reshape(-1)
            coefficients[:,i] = a_hat
            i += 1

        self.coefficients: np.array = coefficients
        return self.coefficients



    def get_errors(self, data=None, p=None) -> tuple[np.array, np.array]:

        '''
        
        After fitting an AR(p) this function returns an array of dimension steps x paths containing errors defined as y - y_hat

        By default data is the one generated through specific function, however if the function fit_ar() has been provided by other data, it is possible to pass it.
        (same for p)

        '''

        if data is None:
            data = self.data
        if p is None:
            p = self.p
        coefficients = self.coefficients

        # Preparing y_hat
        steps, _ = data.shape
        y_hat = np.zeros_like(data)
        for i in range(0,p):
            y_hat[i,:] = data[i,:]  # First p rows filled with initial values

        # Generate the processes of y_hat with the coefficients given by fit_ar()
        if coefficients.ndim == 1:
            a_0 = coefficients[0]
            a = coefficients[1:].reshape(p, 1)    
        else:
            a_0 = coefficients[0, :]           # (paths,)
            a = coefficients[1:, :]            # (p,paths)

        for i in trange(p, steps):
            window = y_hat[i-p:i, :]  # (p,paths)    
            y_hat[i, :] = a_0 + np.sum(a * window, axis=0)

        self.eta     = data - y_hat         # errors
        self.epsilon = self.eta / np.std(self.eta, axis=0, keepdims=True) # std errors                 
        return self.epsilon, self.eta



    def jb_test(self, data=None) -> df:

        '''
        
        Gives a df containing stat test e p-value for each path

        '''

        if data is None:
            data = self.data

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
            p_value = 1.0 - chi2.cdf(jb_stat, df=2)

            jb_summary[0,i] = jb_stat
            jb_summary[1,i] = p_value

        return df(jb_summary).rename(index={0:'jb stat', 1:'p value'})



    def auto_correlation_function(self, p, data=None) -> df:

        '''
        
        Computes the acf up to lag p for every path. 
        
        '''

        if data is None:
            data = self.data

        steps, paths = data.shape
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



    def plot_acf(self, acf_summary: df) -> None:

        '''
        
        Plots the acf function
        
        '''

        steps, paths = self.data.shape
        acf_summary = acf_summary.to_numpy()
        p = acf_summary.shape[0] - 1

        lags = np.arange(p + 1)
        conf = 1.0 / np.sqrt(steps)

        plt.figure(figsize=(8, 5))

        # Plot ACF for each path
        for i in range(paths):
            plt.plot(lags, acf_summary[:, i], alpha=0.7)

        # Confidence intervals around 0
        plt.axhline(conf, color='red', linestyle='--', linewidth=1)
        plt.axhline(-conf, color='red', linestyle='--', linewidth=1)

        # Axes and labels
        plt.axhline(0, color='black', linewidth=1)
        plt.title("Autocorrelation Function (ACF)")
        plt.xlabel("Lag")
        plt.ylabel("ACF Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    def study_errors(self, p: int, display_results = True, data = None) -> tuple[df, df, df, df]:

        '''
        
        Returns many pandas dataframes for:
        - Moments up to the fourth
        - Jarque-Bera test
        - Autocorrelation Function
        - Descriptive statistics

        Everything iterated for every path

        '''

        if data is None:
            data, _ = self.get_errors()
        
        steps, paths = data.shape
        self.moments = np.zeros((4, paths))


        for i in range(0,paths):
            col = data[:,i]
            mu = np.mean(col)
            var = np.var(col)
            std = np.std(col)             
            z = (col - mu) / std
            skewness = np.mean(z**3)
            kurtosis = np.mean(z**4)

            self.moments[:,i] = np.array([mu, var, skewness, kurtosis]).T

        self.moments = df(self.moments).rename(index={0: 'mean', 1: 'variance', 2: 'skewness', 3: 'kurtosis'})
        self.jb_summary = self.jb_test(data=data)
        self.acf = self.auto_correlation_function(p=p, data=data)

        if display_results:
            print("\n")
            print("="*100)
            print("STUDYING ERRORS")
            print("="*100)
            print("\n")

            print("\n")
            print("="*50)
            print("DESCRIPTIVE STATISTICS")
            print("="*50)
            display(df(self.epsilon).describe())

            print("\n")
            print("="*50)
            print("MOMENTS SUMMARY")
            print("="*50)
            display(self.moments)

            print("\n")
            print("="*50)
            print("JARQUE–BERA NORMALITY TEST RESULTS")
            print("="*50)
            display(self.jb_summary)

            print("\n")
            print("="*50)
            print("AUTOCORRELATION FUNCTION (ACF)")
            print("="*50)
            display(self.acf)

        return self.moments, self.jb_summary, self.acf, df(self.epsilon).describe()
        





### For testing and debugging
if __name__ == "__main__":
    model = AutoRegressive(steps=1_000, paths=10, a=np.array([0.2, 1]), start=0)
    data = model.generate()
    model.plot_paths()
    coefficients = model.fit_ar()
    print(coefficients)           # They should match (on average) the given a
    eps, _ = model.get_errors()   # should be N(0,1) in this example
    moments, jb, acf, stat = model.study_errors(10)
    print(moments)
    print(jb)
    print(acf)
    print(stat)