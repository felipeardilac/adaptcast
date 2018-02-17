# adaptcast  
ADAPTATIVE OPERATOR FORECAST  
The purpose of this lib is to train and evaluate a state space reconstruction adaptive models for time-series prediction.  Using DEAP GA as optimization engine to calibrate the hyperparameters of the model

# EXAMPLE 1: HOW TO RUN AN ADAPTIVE OPERATOR

#Load the data

data = pd.read_csv('D:/TRABAJO/FORECAST/PAICOL.csv', index_col=0, parse_dates=True,usecols =[0,1,2])

#Interpolate the missing values

dataInterp=interpolate(data.values)

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

target,forecast=adaptativeOperator(targetData,inputData,lagConf,window,delta,forecasts)


#PLOT PERFORMANCE

plotPerformance(target,forecast,delta=1)

plt.savefig('grid_figure.png')


# EXAMPLE 2: HOW CALIBRATE THE ADAPTIVE OPERATOR PARAMETERS USING GA 

Please run adaptcastGA.py
