# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:34:45 2018

@author: Felipe
"""

# -*- coding: utf-8 -*-
"""
Feb 2018
ADAPTCAST
FORECAST PAKAGE
felipeardilac@gmail.com

"""
# =============================================================================
# TUNE HYPERPARAMETERS OF ADAPTCAST MODEL
# WITH DEAP GENETIC ALGORITHM  
# =============================================================================

import array
import random
import numpy
import numpy as np
import pandas as pd

import adaptcast as ac

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# =============================================================================
# INITIALIZE DATA
# =============================================================================

np.random.seed(1120)

##Load the data
#data = pd.read_csv('D:/TRABAJO/PRONOSTICO CAUDALES/CODIGO/PYTHON/PAICOL.csv', index_col=0, parse_dates=True,usecols =[0,1,2])
data = pd.read_csv('D:/TRABAJO/PRONOSTICO CAUDALES/CODIGO/PYTHON/COLORADOS.csv', index_col=0, parse_dates=True)

##Fill the missing the data
dataInterp=ac.interpolate(data.values)
data= dataInterp

#Parameters
forecasts=test_length=365*1
delta=1
minWindow=181
maxWindow=365*5
maxLag=30
windowRange=np.round(np.linspace(minWindow, maxWindow, num=1+maxLag)).astype(int)

ngen=20
popSize=300
# I/O
inputData= data
targetData= data[:,1].reshape(-1,1)


# =============================================================================
# OPTIMIZATION OF HYPERPARAETERS
# =============================================================================
gene_length=data.shape[1]+1

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, maxLag)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, gene_length)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# =============================================================================
#OBJECTIVE FUNCTION
# =============================================================================
def decode(chromosome):
    # Decode GA solution to windowSize and lagConf    
    windowSize = windowRange[chromosome[0]]
    lagConf =  np.array( chromosome[1:], dtype=np.int32) 
    return windowSize,lagConf

def objectiveFunction(chromosome):
    # Decode GA solution to windowSize and lagConf  
    #windowSize,lagConf=decode(chromosome)
    windowSize = windowRange[chromosome[0]]
    lagConf =  np.array( chromosome[1:], dtype=np.int32) 
#    print('\nWindow Size: ', windowSize, ', Lag Config: ', lagConf)

    
    # Return fitness score of 100 if window_size or num_unit is zero
    if np.sum(lagConf) == 0:
        cost=10
#        print('Validation s/sd: ',cost,'\n')
        return cost,
    else:
        target,forecast=ac.adaptativeOperator(targetData,inputData,
                                  lagConf,
                                  windowSize,1,
                                  test_length)
    
        cost=ac.ssigmadelta(target,forecast,1)
#        print('Validation s/sd: ', cost,'\n')

    return cost,

toolbox.register("evaluate", objectiveFunction)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    
    
    random.seed(64)
    
    pop = toolbox.population(n=popSize)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, 
                                   stats=stats, halloffame=hof, verbose=True)
    
    fitnesses = list(map(toolbox.evaluate, pop))
    # Print top N solutions - (1st only, for now)
    best_individuals = tools.selBest(pop,k = 1)
    
    
    for bi in best_individuals:
        # Decode GA solution to integer for window_size and num_units
        best_windowSize = windowRange[bi[0]]
        best_lagConf =  np.array( bi[1:], dtype=np.int32) 
        print('\nBest Window Size: ', best_windowSize, ', Best Lag Config: ', best_lagConf)
        target,forecast=ac.adaptativeOperator(targetData,inputData,
                                   best_lagConf,
                                   best_windowSize,1,
                                   test_length)
    cost=ac.ssigmadelta(target,forecast,1)
    print('Validation s/sd: ', cost,'\n')
    plt=ac.plotPerformance(target,forecast,delta=1)
    # plt.savefig('grid_figure.pdf')
    plt.savefig('ModelPerformance.png')
    
    return pop, log, hof

if __name__ == "__main__":
    main()
    
