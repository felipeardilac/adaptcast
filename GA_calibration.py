# -*- coding: utf-8 -*-
"""
Feb 2018
ADAPTCAST
FORECAST PAKAGE
@author: felipeardilac@gmail.com
"""
# =============================================================================
# TUNE HYPERPARAMETERS OF ADAPTCAST MODEL
# WITH DEAP GENETIC ALGORITHM  
# =============================================================================

import array
import random
import numpy
import os.path
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
##########################
# CHANGE THIS DIRECTORY!!
##########################
rootDir="D:/TRABAJO/PRONOSTICO CAUDALES/CODIGO/PYTHON/"
#data = pd.read_csv(PAICOL.csv', index_col=0, parse_dates=True,usecols =[0,1,2])
#data = pd.read_csv('D:/TRABAJO/PRONOSTICO CAUDALES/CODIGO/PYTHON/COLORADOS.csv', index_col=0, parse_dates=True)
#data_df = pd.read_csv(rootDir+"PAICOL.csv", index_col=0, parse_dates=True)
#data_df = pd.read_csv(rootDir+"COLORADOS.csv", index_col=0, parse_dates=True)
data_df = pd.read_csv(rootDir+"FLOWS_MAG.csv", index_col=0, parse_dates=True)

##Fill the missing the data
dataInterp=ac.interpolate(data_df.values)
data= dataInterp

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*
#Parameters
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*
#forecasts=test_length=365
#delta=1
#minWindow=30
#maxWindow=30*5
#maxLag=5

forecasts=test_length=12*3
delta=1
minWindow=12
maxWindow=12*5
maxLag=5

windowRange=np.round(np.linspace(minWindow, maxWindow, num=1+maxLag)).astype(int)

ngen=100
popSize=500
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*
# I/O
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*

TARGET_INDEX=0
inputData= data
target_name=list(data_df)[TARGET_INDEX]
input_names=list(data_df)
print("TARGET : ",target_name)
targetData= data[:,TARGET_INDEX].reshape(-1,1)
#the saved optimization file
savedRun= rootDir+target_name+".csv"

# =============================================================================
# OPTIMIZATION OF HYPERPARAETERS
# =============================================================================
gene_length=data.shape[1]+1

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, maxLag)
#
## Structure initializers
#toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, gene_length)
#toolbox.register("population", tools.initRepeat, list, toolbox.individual)
def initIndividual(icls, content):
    return icls(content)

def initPopulation(pcls, ind_init, filename):

#    the lag configurations as single variables
    singleVariables=np.identity(gene_length-1).astype(int)
    for d in range(2,maxLag):
        singleVariables=np.concatenate((singleVariables,d*np.identity(gene_length-1).astype(int)),axis=0)

#    Add single lags with the average window
    contents = np.concatenate(((np.round(maxLag/2)*np.ones(singleVariables.shape[0])).astype(int).reshape(-1,1),
                              singleVariables),axis=1).astype(int)
 
#    if there is a run already?
    if os.path.isfile(savedRun):
        oldBest = np.genfromtxt(filename, delimiter=',').astype(int)
        contents = np.concatenate((oldBest.reshape(1,-1),contents),axis=0).astype(int)
 #        oldBest= pd.read_csv(savedRun, index_col=0)

    
#    the rest just random
    randomPortion=popSize-contents.shape[0]
    if randomPortion>0:    
        contents = np.concatenate((contents,
        np.random.randint(0, high=maxLag, size=(randomPortion,contents.shape[1]))),
        axis=0).astype(int)
    return pcls(ind_init(c) for c in contents)

# Structure initializers
toolbox.register("individual_guess", initIndividual, creator.Individual)
toolbox.register("population_guess", initPopulation, list, creator.Individual, savedRun)

#toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, gene_length)
#toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#population = toolbox.population(n=popSize)

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
        cost=99999
#        print('Validation s/sd: ',cost,'\n')
        return cost,
    else:
        target,forecast=ac.adaptativeOperator(targetData,inputData,
                                  lagConf,
                                  windowSize,1,
                                  test_length)
    
        cost=ac.ssigmadelta(target,forecast,1)
        if np.isnan(cost) or cost==0:
            bizarro=666
#        print('Validation s/sd: ', cost,'\n')

    return cost,

toolbox.register("evaluate", objectiveFunction)


toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=maxLag, indpb=1/(gene_length+1))
#toolbox.register("select", tools.selBest, k=5)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("migrate", tools.migRing, k=5, selection=tools.selBest,
    replacement=tools.selRandom)

#toolbox.register("mate", tools.cxTwoPoint)
#toolbox.register("mutate", tools.mutFlipBit, indpb=1/gene_length)
##toolbox.register("select", tools.selBest, k=5)
#toolbox.register("select", tools.selTournament, tournsize=5)
#toolbox.register("migrate", tools.migRing, k=5, selection=tools.selBest,
#    replacement=tools.selRandom)
#toolbox.regiter("variaton", algorithms.varAnd, toolbox=toolbox, cxpb=0.7, mutpb=0.3)
def main():
    
    
    random.seed(64)
    
#    pop = toolbox.population(n=popSize)
    pop = toolbox.population_guess()
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
    
    chromosomeNames=input_names
    chromosomeNames.insert(0, "WINDOW")
#    BEST=pd.DataFrame(bi, columns=chromosomeNames)
#    BEST.to_csv(savedRun)
#    bi.tofile(savedRun)
    np.savetxt(savedRun, bi, delimiter=',') 
    return pop, log, hof

if __name__ == "__main__":
    main()
    
