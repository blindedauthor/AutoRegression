"""
Module containing the functions to implement the genetic algorithm

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
import random
import Optimization_function
import numpy as np
#----------------------------------------------------------------------


def mutate_function(individual_original, perc_):
    """
    Function to perform mutation operation
    Input: original individual, percentage of genome (bit string) to mutate
    Output: mutated individual
    """
    # Create random genome of 0s and 1s
    individual_rndm = np.random.randint(
        low=0, high=2, size=len(individual_original))
    # Randomly select the indices, of the bit string genome, to apply the mutation
    num = int(len(individual_original) * perc_)
    ind_ = random.sample(range(0, len(individual_original) - 1), num)
    # Apply random fluctuations on the values of 0s and 1s of the original individual
    # at the random indices selected previously
    for ii in ind_:
        individual_original[ii] = individual_rndm[ii]
    # Return mutated individual
    return individual_original
#----------------------------------------------------------------------


def create_individual(individual_length):
    """
    Function to create individuals of population
    Input: Length of individual
    Output: Individual bit-vector
    """
    # Random bit string (or vector) of pre-determined length
    individual_ = np.random.randint(low=0, high=2, size=individual_length)
    return individual_
#----------------------------------------------------------------------


def getKey(item):
    """
    Function to be used to sort list
    """
    return item[0]
#----------------------------------------------------------------------


def evolve(retain, mutate, pop, x_df, df, obj_str, mult_thread, trgt):
    '''
    Main function of genetic evolution 
    Input: retain fraction of rank selection, mutation probability, population,
    dataframe with independent variables, dataframe with all data, metric to be 
    used ('bic' or 'aic'), multi-thread boolean, dependent variable string
    Output: Evolved population, Best model of optimisation
    '''
    # Calculate fitness (or objective function) with single thread or multi-threading
    if mult_thread:
        import multiprocessing as mp
        job_lst = []
        for _ in pop:
            job_lst.append((_, x_df.copy(), df.copy(), obj_str, trgt))
        p = mp.Pool()
        a = p.imap_unordered(Optimization_function.main_function, job_lst)
        p.close()
        p.join()
    else:
        a = []
        for _ in pop:
            objective_, individ_ = Optimization_function.main_function(
                (_, x_df, df, obj_str, trgt))
            a.append((objective_, individ_))
    # Once fitness function is calculated the individuals are sorted according
    # to fitness metric
    graded = a
    hof = sorted(graded, key=getKey, reverse=False)
    # Retain a fraction ('retain') of population with the higher fitness score
    graded = [x[1] for x in hof]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]
    # Mutate individuals if they are not the best of previous generation
    for individual in parents:
        if mutate > random.random() and not(np.array_equal(hof[0][1], individual)):
            individual = mutate_function(individual, 0.5)
    #----------------------------------------------------------------------
    # Crossover parents to create children
    parents_length = len(parents)
    # Calculate how many children need to be created based on total number of
    # individuals needed, and the parents already available, to complete population
    desired_length = (len(pop) - parents_length)
    # Initiate children list
    children = []
    while len(children) < desired_length:
        # Randomly select indices of two parents
        male = random.randint(0, parents_length - 1)
        female = random.randint(0, parents_length - 1)
        # Proceed if two parenrs are not the same
        if male != female:
            # Select parents
            male = parents[male]
            female = parents[female]
            # Initiate child
            child = [0] * len(male)
            # Allocate parent bits to child with 50% chance from each parent
            # i.e. uniform crossover technique
            for iii in xrange(len(male)):
                if random.random() < 0.5:
                    child[iii] = male[iii]
                else:
                    child[iii] = female[iii]
            children.append(child)
    # Extend parents list to include children
    parents.extend(np.array(children))
    # Return evolved population (parents), best model of optimisation
    return parents, hof[0]


if __name__ == '__main__':
    evolve()
