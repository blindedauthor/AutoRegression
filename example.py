"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/
"""
if __name__ == '__main__':
    #----------------------------------------------------------------------
    # Import necessary modules
    import MainAlgorithm
    import pandas as pd
    import model_fit
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    #----------------------------------------------------------------------
    # Initialise algorithm parameters
    RETAIN = 0.2  # retain fraction from rank selection
    MUTATE = 0.3  # mutation probability
    MAX_GEN = 20  # maximum number of generations of the genetic algorithm
    OPT_METRIC = 'bic'  # use the Bayesian information criterion as the optimisation objective
    MIN_GEN = 8  # minimum number of generations
    TOLE_N_GEN = 5  # minimum number of generations without an improvement in the objective
    MULTI_THREAD = True  # boolean to whether or not perform multithreading
    VIF_VALUE = 5.0  # variance inflation factor thershold
    N_GA_POP = 50  # number of individials in population
    N_GA_BOOT = 100  # number of bootstrap iterations
    #----------------------------------------------------------------------
    # Import data and set target variable to desired column
    # downloaded from http://www.gagolewski.com/resources/data/ordinal-regression/
    data = pd.read_csv('winequality-red.csv')
    # Checks for missing values in data
    assert(not(data.isnull().values.any())), 'Missing values in data'
    # Checks if data are numeric
    is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
    assert(all(item == True for item in is_number(data.dtypes))), 'Data not numeric'
    # Set the name of the target or dependent variable
    target = 'response'
    # Prepare list of independent variables
    independ_vars = list(data.columns)
    independ_vars.remove(target)
    # Scale independent variables to mean 0 and variance of 1
    sclr = StandardScaler(copy=True, with_mean=True, with_std=True)
    data[independ_vars] = sclr.fit_transform(data[independ_vars])
    #----------------------------------------------------------------------
    # Run main function of algorithm
    best_model, series_models, series_vars = MainAlgorithm.main_ga(df=data, trgt=target, 
            n_boot=N_GA_BOOT, n_pop=N_GA_POP, ratio_retain=RETAIN, ratio_mut=MUTATE, 
            min_metric=OPT_METRIC, min_gener=MIN_GEN, tol_gen=TOLE_N_GEN, 
            max_gener=MAX_GEN, mult_thrd=MULTI_THREAD, vif_value=VIF_VALUE)
    #----------------------------------------------------------------------
    # Bar plots of highest selected models and variables
    MainAlgorithm.plot_variables(series_vars.head(10), 'GA variable selection')
    MainAlgorithm.plot_models(series_models.head(5), 'GA model selection')
    # %% Get model summary with Wald test p-values and confidence intervals
    model = model_fit.mdl_fit(best_model, data, target, 0.95)
    # Use to save results
#    series_vars.to_csv('Red Wine Quality Variable Rank.csv')
#    series_models.to_csv('Red Wine Quality Models Rank.csv')
#    model.to_csv('Red Wine Quality Model.csv')
    # %% Calculate multi-class ROC AUC as described in:
    # Hand, D.J. & Till, R.J. Machine Learning (2001) 45: 171. doi:10.1023/A:1010920819831
    # Set R formula as string using the final model parameters calculated by the method
    final_model_R = 'as.ordered(' + target + ') ~ ' + "+".join(best_model)
    # Set R function as string to calculate multi-class ROC AUC
    auc_func_R = """
    multi_class_func=function (mydata,formula,target){
            library(VGAM)
            library(HandTill2001)
            
            mydata[[target]]=ordered(mydata[[target]])
            
            mdl=vglm(formula,family=propodds, data=mydata)
            
            predictions.model=predict(mdl,type='response')
            
            multi_class_auc=(auc(multcap(
            response = mydata[[target]],
            predicted = predictions.model)))
            
    return(multi_class_auc)
    }"""
    # Import rpy2 modules
    from rpy2.robjects.packages import STAP
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    # Define R function to be used in Python
    auc_func = STAP(auc_func_R, "auc_func")
    # Transform pandas dataframe to R format
    Rdf = pandas2ri.py2ri(data)
    # Calculate the multi-class ROC AUC
    auc = auc_func.multi_class_func(Rdf, final_model_R, target)[0]
    # Print results
    print(
        'The optimal model results in a multiclass ROC AUC of: {0:4.2f}'.format(auc))
