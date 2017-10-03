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
def mdl_fit(model_vars, df, y_param, ci_level=0.95):
    """
    Function to fit final model and extract modelling statistics
    Input: model variables as a list, dataframe holding all the data, 
    dependent variable, confidence level for reporting statistics i.e. 0.95 for 95% 
    Output: dataframe with model coefficients and statistics   
    """
    #----------------------------------------------------------------------
    # Import necessary modules
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    import numpy as np
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    from rpy2.robjects.packages import STAP
    import scipy.stats as stats
    #----------------------------------------------------------------------
    # Fit R model
    # Set R function as string to fit model and return results
    string_ord_mdl = """
    mdl_func <- function(formula,df) {
    	library(VGAM)
    	mdl1=vglm(formula,family=propodds, data=df)
    
    	ll=logLik(mdl1)
        coefficients_df=coef(summary(mdl1))
        coefficient_cols=colnames(coefficients_df)
        coefficient_rows=rownames(coefficients_df)
    	output<-list(ll,coefficients_df,coefficient_cols,coefficient_rows)
        return(output)
    }
        """
    # Transform pandas dataframe to R format
    rdf = pandas2ri.py2ri(df)
    # Set R formula as string using the model parameters and dependent variable
    formula = 'as.ordered(' + y_param + ') ~ ' + "+".join(model_vars)
    # Define R function to be used in Python
    ord_ll = STAP(string_ord_mdl, "ord_ll")
    # Fit model
    output_R = ord_ll.mdl_func(formula, rdf)
    # Extract data and place them in Pandas dataframe
    coeff_df_temp = output_R[1]
    coeff_df = pandas2ri.ri2py_dataframe(coeff_df_temp)
    cols_df = list(output_R[2])
    rows_df = list(output_R[3])
    coeff_df.columns = cols_df
    coeff_df.index = rows_df
    #----------------------------------------------------------------------
    # Calculate statistics
    # Number of parameters
    n_vars = len(coeff_df)
    # Degrees for freedom for t-distribution
    deg_free = len(df) - n_vars
    # Calculate alpha value from confidence interval
    alpha_ = 1.0 - ci_level
    # array to hold the low % confidence intervals
    low_arr = np.zeros(len(coeff_df))
    # array to hold the high % confidence intervals
    high_arr = np.zeros(len(coeff_df))
    # array to hold the Wald test p-values
    p_val_arr = np.zeros(len(coeff_df))
    # array to hold the t statistic
    t_value_arr = np.zeros(len(coeff_df))
    # loop counter variable
    index_arr = 0
    for index, row in coeff_df.iterrows():
        # Get standard error for variable coefficient from R model fit data
        std_error = row['Std. Error']
        # Get variable coefficient value from R model fit data
        coeff_value = row['Estimate']
        # Calculate t_critical statistic for desired confidence interval
        t_critical = stats.t.ppf(1 - (alpha_ / 2.), df=deg_free)
        # Calculate low - high confidence interval limits
        low_arr[index_arr] = coeff_value - (t_critical * std_error)
        high_arr[index_arr] = coeff_value + (t_critical * std_error)
        # t statistic calculation to get p-value
        t_value = coeff_value / std_error
        t_value_arr[index_arr] = t_value
        # Calculate p-value
        p_val_arr[index_arr] = 2.0 * \
            (1.0 - stats.t.cdf(np.abs(t_value), deg_free))
        index_arr += 1
    # Set arrays to dataframe columns
    coeff_df['Low ' + str((1.0 - alpha_) * 100) + '%'] = low_arr
    coeff_df['High ' + str((1.0 - alpha_) * 100) + '%'] = high_arr
    coeff_df['P Value'] = p_val_arr
    coeff_df['t Value'] = t_value_arr
    # Delete statistics of R model fit referring to normal distribution 
    coeff_df.drop(['z value', 'Pr(>|z|)'], axis=1, inplace=True)
    # Return dataframe with model fit coefficients and statistics
    return coeff_df
