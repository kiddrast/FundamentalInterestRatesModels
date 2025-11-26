import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt # TODO: maybe plotly looks better
from tqdm import trange
import scipy.stats as stats
from IPython.display import display
import utilities as ut

class AutoRegressive():

    def __init__(self, steps: int, paths: int, a=np.array, start=0, dist='normal', error_var=1, df=None, wald_mean=1):

        self.steps     = steps
        self.paths     = paths
        self.a         = a
        self.p         = a.size - 1
        self.start     = start      # TODO: add the option to give different starting points for each level
        self.start_row = np.full(shape=(1,paths), fill_value=self.start)
        self.dist      = dist
        self.error_var = error_var
        self.df        = df         # Degree of freedom of t
        self.wald_mean = wald_mean  # Mean of inverse normal



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
        elif self.dist == 'wald':
            if self.wald_mean <=0:
                raise('The mean of a wald distribution must be greater than 0!')
            epsilon = random_generator.wald(self.error_var, self.error_var, size=data.shape)

        # Fill data
        for i in trange(self.p, self.steps):
            data[i,:] = self.a[0] + self.a[1:].T @ data[i-self.p:i,:][::-1, :] + epsilon[i,:]
            

        print(f'{self.paths} different AR({self.p}) processes of {self.steps - self.p + 2} steps have been generated with increments following {self.dist} distribution') 

        self.data: np.array = data
        return self.data
    


    def prices_to_log_returns(self, data = None) -> np.array:

        '''
        
        Trasforms the prices into log returns. It must be provided of strictly positive data.
        
        '''

        if data is None:
            data = self.data

        return np.diff(np.log(data))
    


    def returns_to_log_returns(self, data = None) -> np.array:

        '''
        
        Trasforms the simple returns into log returns. It must be provided of strictly positive data (Prices)
        
        '''

        if data is None:
            data = self.data

        return np.log(data)



    def plot_paths(self, data=None, size=(11,3)) -> None:

        if data is None:
            data = self.data

        plt.figure(figsize=size)
        plt.plot(data)
        plt.title(f'AR({self.p}) processes')
        plt.grid(True)
        plt.show()
    


    def fit_ar(self, p=None, data=None, method='ols') -> np.array:  
        
        '''

        If the method si ols it gives an array of dimension len(self.a) x paths of coefficients
        By default data is the one generated through specific function, however if the function fit_ar() has been provided by other data, it is possible to pass it.
        (same for p)

        TODO: add Maximum likelihood method

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



    def get_residuals(self, data=None, p=None) -> tuple[np.array, np.array]:

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

        self.eta     = data - y_hat         # residuals
        self.epsilon = self.eta / np.std(self.eta, axis=0, keepdims=True) # std residuals               
        return self.epsilon, self.eta



    def study_residuals(self, display_results: bool = True) -> None:

        '''
        
        Returns many pandas dataframes for:
        - Moments up to the fourth
        - Jarque-Bera test
        - Autocorrelation Function
        - Descriptive statistics

        '''

        self.residuals_stats = df(self.epsilon).describe()
        self.moments = ut.compute_moments(self.epsilon)
        self.jb_summary = ut.jb_test(self.epsilon)
        self.acf = ut.auto_correlation_function(self.epsilon, 20)

        if display_results:
            print("\n")
            print("="*100)
            print("RESIDUALS DIAGNISTIC")
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
            ut.plot_acf(self.acf, 20)

            print("\n")
            print("="*50)
            print("QQ Plots")
            print("="*50)
            ut.qq_plot(self.epsilon)         

        return 
        





### For testing and debugging
if __name__ == "__main__":

    model = AutoRegressive(steps=1_000, paths=6, a=np.array([0.2, 0.4]), start=100)
    data = model.generate()
    model.plot_paths()

    coefficients = model.fit_ar()
    print(coefficients)              # They should match (on average) the given a
 
    eps, eta = model.get_residuals()
    model.study_residuals()

