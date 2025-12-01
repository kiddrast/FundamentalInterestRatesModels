# AR Analysis Workflow

This repository contains a **Jupyter Notebook** (`AR_analysis.ipynb`) and a set of custom Python functions in the `functions` folder for **autoregressive (AR) time series analysis**. The workflow implements almost everything from scratch using **NumPy, Pandas, Matplotlib, and SciPy (optimize)**.

---

## Notebook: `AR_analysis.ipynb`

The notebook provides a complete **AR(p) modeling workflow**:

1. **Data Generation**
   - Select paths, steps and distribution

2. **AR(p) Process Overview**
   - Definition of AR(p) models
   - White noise innovations \(\varepsilon_t \sim WN(0, \sigma^2)\)
   - Consequences for mean, variance, and autocovariance

3. **Stationarity & Autocorrelation**
   - Characteristic polynomial roots
   - Consequences for mean, variance, and autocovariance for stationarity
   - Theoretical vs sample autocovariance and autocorrelation

4. **Fitting AR(p) Models**
   - **OLS estimation**
   - **Maximum Likelihood Estimation (MLE)**
     - Gaussian innovations
     - Student-t innovations
     - Numerical optimization using `scipy.optimize.minimize`

5. **Residual Analysis**
   - Compute residuals: \(\hat{\epsilon}_t = y_t - \hat{y}_t\)
   - Check distribution: histogram, QQ-plot
   - Check moments: mean ≈ 0, variance ≈ σ²
   - Check autocorrelation (ACF) of residuals

6. **Model Selection**
   - Compute **AIC and BIC** from negative log-likelihood
   - Compare Gaussian vs Student-t AR models



## Functions Folder: `functions/`

Contains custom Python scripts with helper functions used in the notebook:

- Implementations of:
  - AR(p) generation
  - OLS fitting
  - MLE fitting for Gaussian and Student-t
  - Residual calculation and plotting
  - AIC/BIC calculation
  - Autocovariance and autocorrelation (theoretical and sample)
  - ...
    
- Only depends on **NumPy, Pandas, Matplotlib, and SciPy** (optimize module for numerical MLE)

---

## Requirements

- Python ≥ 3.10
- Packages:
  - numpy
  - pandas
  - matplotlib
  - scipy
