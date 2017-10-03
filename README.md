# Summary

This repository holds code for automatic regression modelling, specifically ordinal logistic regression.  
The aim is to create explanatory models with significant variables as defined by the p-values and confidence intervals of their regression coefficients.  

# Setup
### Summary of set up
To run the code you will need both Python and R installed
### Configuration
To run the code you will need Python 2.7.13 (64bit) with the following modules installed:  
pandas 0.19.2  
scikit-learn 0.17.1  
numpy 1.11.3  
rpy2 2.7.8  
matplotlib 2.0.0  
tqdm 4.11.2  
statsmodels 0.6.1  
scipy 0.18.1  

You will also need R installed:  
R 3.3.1
with the 'VGAM' and 'HandTill2001' libraries installed

### rpy2 installation
I recommend installing rpy2 from http://www.lfd.uci.edu/~gohlke/pythonlibs/#rpy2  
Select the rpy2‑2.7.8‑cp27‑none‑win_amd64.whl file to install  
Also you will need to add these paths (environment variables) to your user account  
For example: (change Values to match your configuration)  
Variable: R_HOME Values: C:\Program Files\R\R-3.3.1  
Variable: R_USER Values: C:\Users\username\Anaconda2\Lib\site-packages\rpy2  
If you don't have admin access to your PC type "Edit environment variables for your account" into the Windows start menu to change your user account paths without admin privileges  
Alternatively you can temporarily set paths from within Python using  

```
#!python

import os  
os.environ['R_HOME'] = 'C:/xxxxxxxx/R/R-3.2.2'   
os.environ['R_USER'] = 'C:/xxxxxxxxxxxxxx/Anaconda2/Lib/site-packages/rpy2' 
```
  
you might need to place this command at the top of every .py file that calls rpy2   

# Algorithm
Flowchart of the algorithm with some default values:  
![AutoModel.png](https://bitbucket.org/repo/yp5MdKd/images/4102067091-AutoModel.png)  

The main routine is run as follows:  
```
#!python
best_model, series_models, series_vars = MainAlgorithm.main_ga(df=data, trgt=target, n_boot=N_GA_BOOT, n_pop=N_GA_POP, ratio_retain=RETAIN, ratio_mut=MUTATE, min_metric=OPT_METRIC, min_gener=MIN_GEN, tol_gen=TOLE_N_GEN, max_gener=MAX_GEN, mult_thrd=MULTI_THREAD, vif_value=VIF_VALUE)
```  
The output is best_model: python list with the best model parameters from the algorithm, series_models: a pandas series with all the models generated in order of percentage of selection, series_vars: a pandas series with all the variables selected by all the models in order of percentage of selection  

It is recommended to run example.py to test if the code works and better understand how it runs by looking at the comments

# Example
The ordinal response dataset on wine quality was downloaded from http://www.gagolewski.com/resources/data/ordinal-regression/winequality-red.csv and tested for the example shown in 'example.py'  

The resulting model was:  
![Model coefficients.png](https://bitbucket.org/repo/yp5MdKd/images/2551485050-Model%20coefficients.png)
  
As you can see the automatically generated model has only significant variables (p-value<0.05)

Also a list of the percentage the variables were selected during the algorithm run was produced:  
![Variables rank.png](https://bitbucket.org/repo/yp5MdKd/images/1839157875-Variables%20rank.png)

As well as the order of percentage selection of the models:  
![Models rank.png](https://bitbucket.org/repo/yp5MdKd/images/1042504757-Models%20rank.png)

