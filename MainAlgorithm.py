"""
Module that implements the overall algorithm

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
# Import necessary modules
import pandas as pd
import GeneticAlgorithm
import numpy as np
import vif_functions
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
from collections import Counter
#----------------------------------------------------------------------


def func_output(ga_count_series, boot_iterations):
    """
    Function to return ordered list of model and variable selection frequency
    Input: Series holding counts of all models, bootstrap iterations
    Output: Series holding percent counts of models,
    Series holding percent counts of variables
    """
    all_models = ([list(x) for x in ga_count_series.index])
    df_final = pd.DataFrame()
    df_final['models'] = all_models
    df_final['frequency'] = ga_count_series.values

    list_models = list(df_final['models'])
    list_models_ = [item for sublist in list_models for item in sublist]
    dict_vars = Counter(list_models_)
    series_vars = pd.Series(dict_vars, name='Counts')
    series_vars.sort_values(axis=0, inplace=True, ascending=False)
    series_vars = series_vars.apply(normalise_counts, args=(boot_iterations,))
    series_vars = series_vars.rename('Selection Frequency [%]', inplace=True)

    series_models = df_final.T.squeeze()
    series_models.columns = series_models.iloc[0]
    series_models.drop(series_models.index[0], inplace=True)
    series_models = series_models.iloc[0]
    series_models = series_models.rename(
        'Selection Frequency [%]', inplace=True)
    series_models.sort_values(axis=0, inplace=True, ascending=False)
    series_models = series_models.apply(
        normalise_counts, args=(boot_iterations,))

    return series_models, series_vars
#----------------------------------------------------------------------


def normalise_counts(x, n_bt):
    """
    Function to normalise selection counts to percentage
    Input: raw counts, bootstrap iterations
    Output: normalised percentage of counts
    """
    return (x / float(n_bt)) * 100.0
#----------------------------------------------------------------------


def plot_variables(s, tlt):
    """
    Function to plots variable selection
    Input: variables series object, title of plot
    Output: plot figure 
    """
    s_plot = s.copy()
    s_plot.sort_values(axis=0, inplace=True, ascending=True)
    plt.figure()
    ax = s_plot.plot(kind='barh')
    ax.set_xlabel("Selection Frequency [%]")
    ax.set_title(tlt)
    plt.tight_layout()
#----------------------------------------------------------------------


def plot_models(s1, tlt1):
    """
    Function to plots model selection
    Input: models series object, title of plot
    Output: plot figure 
    """
    s_plot_new = s1.copy()
    s_plot_new.sort_values(axis=0, inplace=True, ascending=True)
    plt.figure(figsize=(14, 8))
    ax = s_plot_new.plot(kind='barh')
    ax.set_xlabel("Selection Frequency [%]")
    ax.set_title(tlt1)
    plt.gcf().subplots_adjust(left=0.3, right=0.99, top=0.9)
    ax.yaxis.label.set_visible(False)
#----------------------------------------------------------------------


def getKey(item):
    """
    Function used in ordering of list
    """
    return item[0]
#----------------------------------------------------------------------


def start_evolve(retain_, mutate_, pop_, x_, y_, obj_str, multithread, trgt):
    '''
    Main function of genetic evolution 
    Input: retain fraction of rank selection, mutation probability, population,
    dataframe with independent variables, dataframe with all data, metric to be 
    used ('bic' or 'aic'), multi-thread boolean, dependent variable string
    Output: Evolved population, Best model of optimisation
    '''
    p_, hof_ = GeneticAlgorithm.evolve(
        retain_, mutate_, pop_, x_, y_, obj_str, multithread, trgt)
    return p_, hof_
#----------------------------------------------------------------------


def main_ga(df, trgt, n_boot=100, n_pop=50, ratio_retain=0.2, ratio_mut=0.3, min_metric='bic',
            min_gener=8, tol_gen=5, max_gener=20, mult_thrd=True, vif_value=5.0):
    '''
    Main function of algorithm
    Input: dataframe holiding all data, string of dependent variable,
    bootstrap iterations, number of individuals,
    retain fraction of rank selection, mutation probability, metric to be 
    used ('bic' or 'aic') in optimisation, minimum number of generations, minimum number of 
    generations with no improvement in best model, maximum number of generations,
    multithread enable boolean, VIF threshold
    Output: best model, final model, variables series  
    '''
    # Import tqdm module to handle progress indication
    import tqdm
    models_lst = []

    # Start algorithm with bootstrap iterations
    for iii in tqdm.tqdm(range(n_boot), desc='Genetic Algorithm Model Selection'):
        # Take bootstrap sample
        file_nm_copy = df.copy()
        file_nm_boot = file_nm_copy.sample(frac=1, replace=True)
        # Reset dataframe index
        file_nm_boot = file_nm_boot.reset_index(drop=True)
        # Drop target (dependent) variable from X dataframe
        X_boot = file_nm_boot.drop([trgt], axis=1)
        # Y datafram holds all variables including the target
        # to be used in the ordinal regression fit in R
        Y = file_nm_boot
        # Remove collinearity
        X, col_lst_col = vif_functions.rmv_multicol(X_boot, vif_value)
        # Calculate number of individials, i.e. number of parameters remainig
        # after collinearity removal
        n_individual = len(X.columns)
        # Create initial population of genetic algorithm
        pop = [GeneticAlgorithm.create_individual(
            n_individual) for _ in range(n_pop)]
        # Initial number of generations (n_gen) is zero
        n_gen = 0
        # List to hold best model of each iteration
        hof_lst = []
        # List to hold BIC or AIC values of best individual for each generation
        track_obj_vals = []
        # Start genetic algorithm with condition of exceeding maximum number
        # of generations (max_gener)
        while n_gen < max_gener:
            # The genetic algorithm returns the current population (pop_temp)
            # and the best model (hof)
            pop_temp, hof = start_evolve(ratio_retain, ratio_mut, pop,
                                         X, Y, min_metric, mult_thrd, trgt)
            # Append BIC or AIC value of best individual
            track_obj_vals.append(hof[0])
            # Append best individual
            hof_lst.append(hof)
            # Set next population to the current one returned from the genetic algorithm
            pop = pop_temp[:]
            # Increment the generation index
            n_gen += 1
            # Check if number of minimum generations is reached (n_gen>min_gener)
            # and if we have a minimum number of generations (tol_gen) without
            # a change in the best model
            if len(set(track_obj_vals[-tol_gen:])) == 1 and n_gen > min_gener:
                break
        # Sort best models according to their BIC (or AIC)
        hof_sort = sorted(hof_lst, key=getKey, reverse=False)
        # Select binary vector of model to be used to select the final
        # model parameters
        gen_arr = np.array(hof_sort[0][1])
        # Apply binary vector on the total model parameters in dataframe X
        # to get the final parameters by column name
        vars_lst = list(X.columns)
        gen_ind_0 = np.where(gen_arr == 0)
        gen_ind_0 = gen_ind_0[0]
        vars_lst2 = vars_lst[:]
        for i in sorted(gen_ind_0, reverse=True):
            del vars_lst2[i]
        final_vars_ga = vars_lst2[:]
        # Append parameters in list
        models_lst.append(final_vars_ga)
    # Count unique models
    aa = Counter(map(frozenset, models_lst))
    df = pd.DataFrame.from_dict(aa, orient='index')
    count_series = df.ix[:, 0]
    count_series = count_series.copy()
    count_series.sort_values(inplace=True, ascending=False)
    # Select best model of all the models
    best_model = list(list(count_series.index)[0])
    # Get the final model and variables by selection frequency as
    # Pandas series objects
    series_models, series_vars = func_output(count_series, n_boot)
    # Return best model, final models and variables series
    return best_model, series_models, series_vars
