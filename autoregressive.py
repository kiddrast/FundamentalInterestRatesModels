import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # TODO: maybe plotly looks better
from tqdm import trange


class AutoRegressive:


    def __init__(self, steps: int, paths: int, a=np.array, start=0, dist='normal', error_var=1, df=None, wald_mean=1):

        self.steps     = steps
        self.paths     = paths
        self.a         = a
        self.p         = a.size - 1
        self.start     = start      # TODO: add the option to give different starting points for each level
        self.start_row = np.full(shape=(1,paths), fill_value=self.start)
        self.dist      = dist
        self.error_var = 1
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
            data[i,:] = self.a[0] + self.a[1:].T @ data[i-self.p:i,:] + epsilon[i,:]

        print(f'{self.paths} different AR({self.p}) processes of {self.steps - self.p + 2} steps have been generated with increments following {self.dist} distribution') 

        self.data: np.array = data
        return self.data


    def plot_paths(self, size=(11,3), title=None):
        if title is None:
            title = f'AR({self.p}) processes'
        plt.figure(figsize=size)
        plt.plot(self.data)
        plt.title(title)
        plt.grid(True)
        plt.show()


    def fit_ar(self, p=None, data=None, method='ols') -> np.array:  
        
        '''

        If the method si ols it gives an array of dimension len(self.a) x paths of coefficients

        TODO: add Maximum likelihood method

        '''
        
        if p == None:
            p = self.p
        if data == None:
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

        self.coefficients = coefficients
        return self.coefficients


    def get_errors(self, data=None, p=None) -> np.array:

        '''
        
        After fitting an AR(p) this function returns an array of dimension steps x paths containing errors defined as y - y_hat

        By default data is the one generated through specific function, however if the function fit_ar() has been provided by other data, it is possible to pass it.
        (same for p)

        '''

        if data == None:
            data = self.data
        if p == None:
            p = self.p

        # Preparing y_hat
        steps = data.size[0]
        y_hat = np.zeros_like(data)
        for i in range(0,p):
            y_hat[i,:] = data[i,:]  

        # Generate the processes of y_hat with the found coefficients 
        i=0
        for col in y_hat.T:
            a = self.coefficients[:,1]
            for i in trange(p, steps):
                col[i] = a[0] + a[1:].T @ y_hat[i-p:i,:]
            i+=1
        self.epsilon = data - y_hat[1,:]
        return self.epsilon


    '''   
        TODO
        def study errors:

    '''

### For testing and debugging
if __name__ == "__main__":
    model = AutoRegressive(steps=1_000, paths=1, a=np.array([0.2,0.5,-0.4]), start=0)
    data = model.generate()
    model.plot_paths()
    coefficients = model.fit_ar()
    print(coefficients) # They should match (on average) the given a
    model.get_errors()