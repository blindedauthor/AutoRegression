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
def main_function((individ, X_df, df, objective_str, trgt)):
    """
    Function to calculate BIC or AIC of model
    Input: individual defining model parameters to be used, 
    dataframe with independent variables, dataframe with all data,
    metric to be used (BIC or AIC), dependent variable string
    Output: mutate individual
    """
    #----------------------------------------------------------------------
    # Import necessary modules
    import numpy as np
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    from rpy2.robjects.packages import STAP
    #----------------------------------------------------------------------
    # Remove variables from dataframe that are not part of the individual's genome
    vars_lst = list(X_df.columns)   # Get list of the available variables
    gen_ind_0 = np.where(individ == 0)  # Find which elements are set to 0
    gen_ind_0 = gen_ind_0[0]  # Get indices of variables to be removed
    vars_lst2 = vars_lst[:]  # Copy list of all variables
    # Remove variables
    for i in sorted(gen_ind_0, reverse=True):
        del vars_lst2[i]
    #----------------------------------------------------------------------
    # Fit model in R
    # Create formula to be used in R 
    myString = "+".join(vars_lst2)
    stable_str = 'as.ordered(' + trgt + ') ~ '
    formula = stable_str + myString
    # Transform Pandas dataframe to R
    rdf = pandas2ri.py2ri(df)
    # Define R function as string
    string = """
    mdl_func <- function(formula,df) {
            library(VGAM)
            mdl1=vglm(formula,family=propodds, data=df)  
            ll=logLik(mdl1)    	
        return(ll)
    }
    """
    ord_ll = STAP(string, "ord_ll")
    # Calculate AIC and BIC based on LogLikelihood (ll_)
    try:
        ll_ = ord_ll.mdl_func(formula, rdf)
        ll_ = ll_[0]
    # In case LogLikelihood calculation fails
    except:
        ll_ = -1000.0
    k = float(len(vars_lst2))
    n = float(len(df))
    aic_ = (2.0 * k) - (2 * ll_)
    bic_ = (np.log(n) * k) - (2 * ll_)
    # Return AIC or BIC depending on used choice
    if objective_str == 'aic':
        obj_ = aic_
    elif objective_str == 'bic':
        obj_ = bic_
    else:
        obj_ = np.nan
    # Return optimisation metric
    return obj_, individ
