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
#----------------------------------------------------------------------
# Function to calculate VIF taken from statsmodels with the addition to
# handle divisions with zero and nan's


def variance_inflation_factor(exog, exog_idx):
    import numpy as np
    from statsmodels.regression.linear_model import OLS
    '''variance inflation factor, VIF, for one exogenous variable

    The variance inflation factor is a measure for the increase of the
    variance of the parameter estimates if an additional variable, given by
    exog_idx is added to the linear regression. It is a measure for
    multicollinearity of the design matrix, exog.

    One recommendation is that if VIF is greater than 5, then the explanatory
    variable given by exog_idx is highly collinear with the other explanatory
    variables, and the parameter estimates will have large standard errors
    because of this.

    Parameters
    ----------
    exog : ndarray, (nobs, k_vars)
        design matrix with all explanatory variables, as for example used in
        regression
    exog_idx : int
        index of the exogenous variable in the columns of exog

    Returns
    -------
    vif : float
        variance inflation factor

    Notes
    -----
    This function does not save the auxiliary regression.

    See Also
    --------
    xxx : class for regression diagnostics  TODO: doesn't exist yet

    References
    ----------
    http://en.wikipedia.org/wiki/Variance_inflation_factor

    '''
    k_vars = exog.shape[1]
    x_i = exog[:, exog_idx]
    mask = np.arange(k_vars) != exog_idx
    x_noti = exog[:, mask]
    r_squared_i = OLS(x_i, x_noti).fit().rsquared
    if (r_squared_i == 1.0 or np.isnan(r_squared_i)):
        vif = 100.0
    else:
        vif = 1. / (1. - r_squared_i)
    return vif
#----------------------------------------------------------------------


def rmv_multicol(x_df, thres):
    """
    Function to remove multicollinearity based on VIF
    Input: dataframe with independent variables, VIF threshold
    Output: the dataframe after removal of collinear variables, 
    the variable names that were removed   
    """
    #----------------------------------------------------------------------
    # Import necessary modules
    import numpy as np
    np.seterr(divide='ignore', invalid='ignore')
    # Copy dataframe
    x = x_df.copy(deep=True)
    # Remove nan's from dataframe
    x = x.dropna()
    # Flag variable to break execution of loop
    flag = 0
    # List to keep values of variables to be removed
    drop_var_ = []
    # Keep track of iterations to avoid exceeding the number of variables
    count_iters = 0
    # While x not empty (i.e. we removed all parameters) and infinite iteration
    # flag is zero (false)
    while flag == 0 and not x.empty:
        # Transfrom dataframe to array
        X_arr_temp = np.array(x)
        # List to keep values of VIF values
        lst_val = []
        # Calculate all VIFs
        for i in xrange(len(X_arr_temp[0, :])):
            lst_val.append(variance_inflation_factor(X_arr_temp, i))
        # Find variable with maximum VIF value
        inde_max = np.argmax(np.array(lst_val))
        val_max = np.max(np.array(lst_val))
        # If variable VIF is more than threshold remove that variable
        if val_max > thres:
            drop_var_.append(x.columns[inde_max])
            x.drop(x.columns[inde_max], axis=1, inplace=True)
        # check in case we removed all variables and exit loop (flag=1)
        elif val_max <= thres or x.empty:
            flag = 1
        # If we looped more than total number of variables something went wrong
        # so exit loop (flag=1)
        if count_iters > len(x_df.columns):
            flag = 1
        # Increment loop counter
        count_iters += 1
    # Return the dataframe after removal of collinear variables
    remainig_nms = list(x.columns)
    x_return = x_df[remainig_nms]
    # Return 1) the dataframe after removal of collinear variables (x_return)
    # 2) the variable names that were removed
    return x_return, drop_var_
