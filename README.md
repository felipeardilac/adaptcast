# adaptcast  
# ADAPTATIVE OPERATOR FORECAST  
The purpose of this lib is to train and evaluate a state space reconstruction adaptive models for time-series prediction.  Using DEAP GA as optimization engine to calibrate the hyperparameters of the model

# EXAMPLE 1: HOW TO RUN AN ADAPTIVE OPERATOR
First you have to download 
numpy, pandas, matplotlib, sklearn, DEAP
```
>pip install --upgrade numpy
>pip install --upgrade scipy
>pip install --upgrade pandas
>pip install --upgrade matplotlib
>pip install --upgrade sklearn
>pip install --upgrade DEAP
```
and adaptcast
```
>pip install --upgrade https://github.com/felipeardilac/adaptcast
```
Then please download the data [PAICOL](adaptcast/PAICOL.csv) and run the following code:
```
import adaptcast as ac

#Load the data
rootDir='D:/...your dir'
data = pd.read_csv(rootDir+'/PAICOL.csv', index_col=0, parse_dates=True,usecols =[0,1,2])

#Interpolate the missing values
dataInterp=ac.interpolate(data.values)
data= dataInterp

#Define the parameters
lagConf=np.asarray([1,4])
window=600
forecasts=365*1
delta=1 #Just 1 for now

#I/O
inputData= data
targetData= data[:,0] #Choose the target as the first row

#APPLY OPERATOR
target,forecast=ac.adaptativeOperator(targetData,inputData,lagConf,window,delta,forecasts)

#PLOT PERFORMANCE
ac.plotPerformance(target,forecast,delta=1)
#SAVE PERFORMANCE
plt.savefig('grid_figure.png')
```

# EXAMPLE 2: HOW CALIBRATE THE ADAPTIVE OPERATOR PARAMETERS USING GA 

Please download the data [FLOWS_MAG](adaptcast/FLOWS_MAG.csv) adn run the following code [GA_calibration.py](adaptcast/GA_calibration.py)

This code had been tested on python 3.5





