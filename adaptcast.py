# -*- coding: utf-8 -*-
"""
Feb 2018
ADAPTCAST
FORECAST PAKAGE
@author: felipeardilac@gmail.com

"""
#Utility libraries
# =============================================================================
#Data utilities
import pandas as pd
import numpy as np
#To plot
# =============================================================================
# from matplotlib import pyplot
# =============================================================================
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as colors
# from pandas.tools.plotting import autocorrelation_plot
# ModeL OPERATOR
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error
#from math import sqrt
# =============================================================================
# Create a multiple lagged state space of the shape
# y(t)= f(y(t-1) y(t-2) x(t) x(t-1) x(t-2))

def shift(data,k):
    x=np.zeros(k, dtype=float)
    x.fill(np.nan)
    return np.hstack((x,data))[0:len(data)].T

def createLags(inputData, lagConfiguration):
    fistCol=1
    laggedData=np.zeros(1, dtype=float)
    laggedData.fill(np.nan)
    for i in range(0, lagConfiguration.shape[0]):
        lags=lagConfiguration[i]
        if lags!=0: 
            for j in range(1, lags+1):
                #names.append(list(data)[i]+" "+str(j))
                if fistCol==1:
                    #laggedData=inputData.iloc[:,i].shift(j).dropna()
                    laggedData=shift(inputData[:,i],j).reshape(-1,1)
                    fistCol=0
                else:
                    #laggedData=pd.concat([laggedData,inputData.iloc[:,i].shift(j)],axis=1).dropna()
                    laggedData=np.append(laggedData,shift(inputData[:,i],j).reshape(-1,1), axis=1)
        
            
    #laggedData.columns=names
    return laggedData


#Calculate the Root mean squared error
def rmse(target,simdata):
    #Length of the data
    n = len(target)
    #Number of NAs in the data
    nans=np.isnan(target) |  np.isnan(simdata)
    nanNumb=sum(nans)
          # RMSE
    cost = np.power(sum(np.power((target[nans==0] - simdata[nans==0]),2)/(n-nanNumb)),1/2)
    return cost
#Calculate the Normalized Root mean squared error
def nrmse(target,simdata):
    #Length of the data
    n = len(target)
    #Number of NAs in the data
    nans=np.isnan(target) |  np.isnan(simdata)
    nanNumb=sum(nans)
          # NRMSE
    cost = np.power(sum(np.power((target[nans==0] - simdata[nans==0]),2)/(n-nanNumb)),1/2)
    cost=cost/np.mean(target[nans==0])
    return cost
#Calculate the Porcentual Root mean squared error
def prmse(target,simdata):
    #Length of the data
    n = len(target)
    #Number of NAs in the data
    nans=np.isnan(target) |  np.isnan(simdata)
    nanNumb=sum(nans)
          # PRMSE
    cost = np.power(sum(np.power((target[nans==0] - simdata[nans==0]),2)/(n-nanNumb)),1/2)
    cost=cost/target[nans==0]
    return cost
#Calculate Nash–Sutcliffe model efficiency coefficient
def nash(target,simdata):
    #Length of the data
    n = len(target)
    #Number of NAs in the data
    nans=np.isnan(target) |  np.isnan(simdata)
    nanNumb=sum(nans)
          # Nash–Sutcliffe
    cost = np.power(sum(np.power((target[nans==0] - simdata[nans==0]),2)/(n-nanNumb)),1/2)
    var=np.sum(np.power((target[nans==0]-np.mean(target[nans==0])),2))
    cost=1-(cost/var)
    return cost
#Calculate the russian hidrological criteria s/sdelta
def ssigmadelta(target,simdata,delta):
    # Estimate the delta
    i=np.arange(1,len(target) - delta)
    deltas= target[i+delta] - target[i]

    # Mean of the deltas
    md = np.nanmean(deltas)
    # SD of the deltas
    sigmadelta = np.nanstd(deltas - md)
    # RMSE
    s=rmse(target,simdata)
    # s = (sum((target - simdata)^2,na.rm = TRUE)/(n-nanVal))^(1/2)
    #S/sigmadelta
    ssd = s/sigmadelta
    cost = ssd
    return cost
#Fast linear regression method
def lmFast(y,x): 
#    add col of 1s
    X = np.concatenate((np.ones((len(x),1), dtype=int), x), axis=1)  
    Y = np.array(y).reshape(-1, 1)
#    take the ones with no nan value
    indexNA=(np.sum(np.isnan(x),axis=1)!=0).reshape(-1, 1) | (np.isnan(y)).reshape(-1, 1)
    indexNA=indexNA.reshape(-1)

    X=X[indexNA==0,:]
    Y=Y[indexNA==0]
    
    coef = np.linalg.solve(X.T.dot(X), X.T.dot(Y))    
#    coef = np.linalg.lstsq(X, Y)[0]
    
#    print("lm_ok : ",np.allclose(np.dot(X.T.dot(X), coef), X.T.dot(Y)))
    

    return coef
#Predict with Fast linear regression method
def predictlmFast(w,x):
  X = np.concatenate((np.ones((len(x),1), dtype=int), x), axis=1)
  Y=w.T.dot(X.T)
  return Y.reshape(-1)
# plot Performance
def plotPerformance(target,prediction,delta=1):
    res=target-prediction
    text1= "RMSE = " + np.array2string(np.round(rmse(target,prediction),  decimals= 4))+" "+"NRMSE = "+ np.array2string(np.round(nrmse(target,prediction), decimals = 4))+" "+"NASH = "+ np.array2string(np.round(nash(target,prediction), decimals = 4))+" "+"S/Sd = "+ np.array2string(np.round(ssigmadelta(target,prediction,delta), decimals = 4))
    text2= "r = " + np.array2string(np.round(np.corrcoef(target[:,0],prediction[:,0])[0,1],  decimals= 4))
    
    # plot it
    t = np.arange(1, len(target)+1).reshape(-1,1)
    
    fig = plt.figure(figsize=(20, 10))    
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1])
    plt.rcParams.update({'font.size': 22})
    ax0 = plt.subplot(gs[0])
    ax0.plot(t,target,marker=".",label='Target')
    ax0.plot(t,prediction,"--",marker=".",color="#B8059A",label='Forecast')
    ax0.set_title(text1)
    ax0.set_xlim([1,np.nanmax(t)])
    ax0.set_ylim([np.nanmin(target),np.nanmax(target)])
    ax0.legend()
    
    
    ax1 = plt.subplot(gs[1])
    ax1.scatter(target, 
                prediction,s=10,
                color="black",marker="o")
    ax1.set_title(text2)
    ax1.set_xlim([np.nanmin(target),np.nanmax(target)])
    ax1.set_ylim([np.nanmin(target),np.nanmax(target)])
    # The regresion fit
    fit=lmFast(target.reshape(-1, 1),prediction.reshape(-1, 1)) 
    y1=predictlmFast(fit,target.reshape(-1, 1))
    ax1.plot(target,y1,"--",color="#8B008B")
    
    

    ax2 = plt.subplot(gs[2])
    #ax2.stem(t,res,color="#05B851",markerfmt=" ")
    ax2.stem(res, markerfmt=' ')
    ax2.set_title("Residuals")
    ax2.set_xlim([1,np.nanmax(t)])
    ax2.set_ylim([np.nanmin(res),np.nanmax(res)])
    
    ax3 = plt.subplot(gs[3])
    n_bins=np.round(np.log(res.shape[0],order=2)+1).astype(int)
#    n_bins=14
    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = ax3.hist(res, bins=n_bins)
        # We'll color code by height, but you could use any scalar
    fracs = N.astype(float) / N.max()
   # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())
    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        #color = plt.cm.viridis(norm(thisfrac))
        color = plt.cm.viridis(norm(thisfrac)*(2/3))
        thispatch.set_facecolor(color)
    ax3.set_title("Hist Residuals")
    
    plt.tight_layout()
    return plt
# Interpolate a dataset using a linear model of
# all variables in the set
def interpolate(data):  
    # Create a copy to interpolate
    InterpData=data
    for i in np.arange(0,data.shape[1]):
        # index of missing values
        indexNA=np.isnan(data[:,i])
        if np.sum(indexNA)!=0:
            # Create a liner model
            y_ok=data[indexNA==0,i]
            x=np.delete(data, np.s_[i], axis=1) 
            x_ok=x[indexNA==0,:]
            x_nan=x[indexNA==1,:]
            fit1=lmFast(y_ok,x_ok)            
            # interpolate
            tInerp1=predictlmFast(fit1, x_nan)
    
            # update the df with the interpolated values
            InterpData[indexNA,i]=tInerp1
    
            tInerp2=InterpData[:,i]
            # THE REST UNABLE TO BE INTERPOLATED
            if np.sum(np.isnan(tInerp2))!=0:
                
                       
                indexNA=np.isnan(tInerp2)
                index=np.arange(0,tInerp2.shape[0]) 
            
                tInerp2[indexNA] =np.interp(index[indexNA==1],index[indexNA==0], tInerp2[indexNA==0])
                # update the df with the interpolated values
                InterpData[:,i]=tInerp2
        
    return InterpData

#AppLy adaptative operator for forecast
def adaptativeOperator(targetData,inputData,lagConf,window,delta,forecasts):
    
    # Parameters
# =============================================================================
#     inputData=input
#     targetData=target
# =============================================================================
    lagConfiguration=lagConf
    windowSize=window
    numberOfForecasts=forecasts
    
    # target+input data
    # (it's not cheating because it use only the lags)
    inputLagedData=createLags(inputData, lagConfiguration)
    
    # target+laged data      
    fullData=np.concatenate((targetData.reshape(-1,1),inputLagedData.reshape(-1,np.nansum(lagConfiguration))), axis=1)
    # Get the lenght of the data    
    dataLength=len(fullData)
    # Get the number of predictors
    numberOfLagedVaribles=inputLagedData.shape[1]
    
    # Check if the number of observations in
    # the calibration window is at least
    # same size+2 of the number of preditor variables
    if numberOfLagedVaribles+1>windowSize:
        #You need a bigger calibration window
        windowSize=numberOfLagedVaribles+2 #At least 2 degrees of freedom
        
    # Extract the validation data set
    #validationData=targetData[seq(dataLength-numberOfForecasts+1,dataLength)]
    validationData=targetData[np.arange(dataLength-numberOfForecasts,dataLength)].reshape(-1,1)
    
    # Create a vector to store the simulations
    simulatedData=np.zeros((numberOfForecasts,1), dtype=float)
    simulatedData.fill(np.nan)    
    
    # Realize the forecast for each calibration window
    for j in range(0,numberOfForecasts):
    
        # select the window for the specific forecast
        windowIndex=np.arange(dataLength-numberOfForecasts+1+j-windowSize-1,
        dataLength-numberOfForecasts+1+j)
        
        
        # Calibration set for the window -1 index
        train=fullData[windowIndex[np.arange(0,len(windowIndex)-1)],:]
        test=fullData[windowIndex[len(windowIndex)-1],:].reshape(1,-1)
        
        y=train[:,0].reshape(-1,1)
        x=train[:,range(1,train.shape[1])] 
        
        
        # Create a liner model
# =============================================================================
        fit=lmFast(y,x)
#        lr = LinearRegression()
#        lr.fit(x,y)
# =============================================================================
        

        # forecast one index outsie of the calibration window set
# =============================================================================
        simulatedData[j,0]=predictlmFast(fit,test[:,range(1,train.shape[1])])
#        simulatedData[j,0]=lr.predict(test[:,range(1,train.shape[1])])

# =============================================================================
        
        
    return   validationData,simulatedData
# =============================================================================
# ======END OF FUNCTIONS=======================================================
# =============================================================================
##Load the data
#data = pd.read_csv('D:/TRABAJO/FORECAST/PAICOL.csv', index_col=0, parse_dates=True,usecols =[0,1,2])
#
#dataInterp=interpolate(data.values)
#data= dataInterp
#lagConf=np.asarray([1,4])
#window=600
#forecasts=365*1
#delta=1
#
#inputData= data
#targetData= data[:,0]
#
## =============================================================================
## APPLY OPERATOR
## =============================================================================
#
#target,forecast=adaptativeOperator(targetData,inputData,lagConf,window,delta,forecasts)
#
#
## =============================================================================
#plotPerformance(target,forecast,delta=1)
## =============================================================================
## plt.savefig('grid_figure.pdf')
## =============================================================================
#plt.savefig('grid_figure.png')
## =============================================================================
