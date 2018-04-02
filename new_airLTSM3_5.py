'''
LSTM RNN using Keras libraries for predicting an individual timeseries based 
on multivariate inputs. Takes data in csv column format (each col is variable 
with first row header). 

Provides user input to:
    - select time series to be predicted, 
    - amount of horizon units to predict, 
    - # of epochs to train, and 
    - # of look-back recurrent cells.

Also asks if the data should be processed - if yes, converts data to a 0-1 scale
based on the column for training. Cyclic data such as wind direction should be
pre-coverted to sin/cosine components prior to loading.

Partitions data into training (80%) and testing (20%) sets.

Takes output data (Y) and leads it for future prediction.
Takes input data (X) and converts into a 3D tensor for Keras based RNN training.
The tensor dimensions are based on (# of samples, # of look_backs, and # of input variables)
A sample is the # of variables x # of look_backs - creating a 2D array. Samples
are prepared by sliding down the list of observations.

X and Y sets (training and test) are adjusted for equal length.
Input nodes = # of input variables, batch training is set for online (1)

The model is trained and MAE calculated for training and test sets. The observed 
and predicted sets are saved in an xlsx file

Brian Freeman 6 Sep 2017
Converted to Python 3.5 on 2 Apr 2018

Updated with comments and plots from Felipe Ukan 17 Sep 2017
'''
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import metrics
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler #, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import sys
from datapreprocess import preprocess   ## local library datapreprocess.py

# convert an array of values into a dataset matrix
def TensorForm(data, look_back):
    #determine number of data samples
    rows_data,cols_data = np.shape(data)
    
    #determine # of batches based on look-back size
    tot_batches = int(rows_data-look_back)+1
    
    #initialize 3D tensor
    threeD = np.zeros(((tot_batches,look_back,cols_data)))
    
    # populate 3D tensor
    for sample_num in range(tot_batches):
        for look_num in range(look_back):
            threeD[sample_num,:,:] = data[sample_num:sample_num+(look_back),:]
    
    return threeD

# fix random seed for reproducibility
np.random.seed(7)

# 
# ***
# 1) load dataset
# ***
data_file_name = 'air_dataset.csv'
df = read_csv(data_file_name, engine='python', skipfooter=3)

a = list(df)  #makes a list of the column names in the dataframe

for i in range (len(a)):    #prints a list of the column names
    print (i, a[i])

last_col = np.shape(df)[1] - 1

# pick column to predict
try:
    target_col = int(input("Select the column number to predict (default = " + a[last_col] + "): "))
except ValueError:
    target_col = last_col   #choose last column as default

# choose look-ahead to predict   
try:
    lead_time =  int(input("How many hours ahead to predict (default = 24)? "))
except ValueError:
    lead_time = 24
    
#convert to floating numpy arrays
dataset1 = df.fillna(0).values
dataset1 = dataset1.astype('float32')
dataplot1 = dataset1[lead_time:,target_col]  #shift training data
dataplot1 = dataplot1.reshape(-1,1)

# normalize the dataset
try:
    process = input("Does the data need to be pre-preprocessed Y/N? (default = y) ")
except ValueError:
    process = 'y'
    
if process == 'Y' or 'y':
    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalerY = MinMaxScaler(feature_range=(0, 1))
    
    dataset = scalerX.fit_transform(dataset1)
    dataplot = scalerY.fit_transform(dataplot1)
    
    print('\nData processed using MinMaxScaler')
else:
    print('\nData not processed')
    
# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# prepare output arrays
# ***
# 2) dataplot[train_size:len(dataset)] changed because it should be dataplot len
# ***
trainY, testY = dataplot[0:train_size], dataplot[train_size:len(dataplot)]

n,p = np.shape(trainY)
if n < p:
    trainY = trainY.T
    testY = testY.T

# resize input sets
trainX1 = train[:len(trainY),]
testX1 = test[:len(testY),]
  
# get number of epochs
try:
    n_epochs = int(input("Number of epochs? (Default = 10)? "))
except ValueError:
    n_epochs = 10
    
# prepare input Tensors
try:
    look_back = int(input("Number of recurrent (look-back) units? (Default = " + str(lead_time+2) + ")? "))
except ValueError:
    look_back = lead_time+2

# mini-batch size    
n_batch = 72
    
# get final approval to compile and train
print('\nInput summary')
print('Loading data from ' + data_file_name)
print('Training on ' + a[target_col])
print('Prediction horizon is ' + str(lead_time))
print('Number of training epochs is ' + str(n_epochs))
print('Number of recurrent units is ' + str(look_back))
print('Number of samples/batch is ' + str(n_batch))

### Make training/testing tensors       
trainX = TensorForm(trainX1, look_back)
testX = TensorForm(testX1, look_back)
print('Number of training samples is ' + str(len(trainX)))
print('Number of test samples is ' + str(len(testX)))

#### check to see if parameters are ok to continue
try:
    contin = input("Continue with model training? (Default = (y)? ")
except ValueError:
    contin = "y"
    
if contin == "n":
    sys.exit()

print('Building model...')
    
# ***
# 3) number of neurons / input_nodes increased for the LSTM layer
# ***
#input_nodes = 50
input_nodes = int(trainX.shape[2] * 2)

# trim target arrays to match input lengths
if len(trainX) < len(trainY):
    trainY = np.asmatrix(trainY[:len(trainX)])
    
if len(testX) < len(testY):
    testY = np.asmatrix(testY[:len(testX)])

model = Sequential()
# ***
# 3) Actual change on the LSTM layer
# ***
model.add(LSTM(input_nodes, activation='sigmoid', recurrent_activation='tanh', 
                input_shape=(trainX.shape[1], trainX.shape[2])))

# add dropout for generalization (default = 0.2)
#model.add(Dropout(0.2)) - can't use dropout with Keras 2.1.2 anymore :( 

# 1 neuron on the output layer
model.add(Dense(1))

# compiles the model
model.compile(loss='mean_squared_error', optimizer='nadam', metrics = [metrics.mae])

# ***
# 4) Increased the batch_size to 72. This improves training performance by more than 50 times
# and loses no accuracy (batch_size does not modify the final result, only how memory is handled)
# ***
#### Start the clock
start1 = time.clock()  

history = model.fit(trainX, trainY, epochs=n_epochs, batch_size=n_batch, validation_data=(testX, testY), shuffle=False)

    # stop clock
end1 = time.clock() 

if (end1-start1 > 60):
    print ("Model trained in {0:.1f} minutes".format((end1-start1)/60.))
else:
    print ("Model trained in {0:.1f} seconds".format((end1-start1)/1.))
# ***
# 5) test loss and training loss graph. It can help understand the optimal epochs size and if the model
# is overfitting or underfitting.
# ***
xhistory = len(history.history['loss'])
xlin = range(1,xhistory+1)
plt.close('all')
plt.plot(xlin,history.history['loss'],color="black", label='Train')
plt.plot(xlin,history.history['val_loss'], color = "black", linestyle = ':', label='Test')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.xticks(np.arange(min(xlin), max(xlin)+1, (max(xlin)+1 - min(xlin))/10))
plt.legend()
plt.show()

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scalerY.inverse_transform(trainPredict)
trainY = scalerY.inverse_transform(trainY)
testPredict = scalerY.inverse_transform(testPredict)
testY = scalerY.inverse_transform(testY)

# ***
# 6) calculate mean absolute error. Different than root mean squared error this one
# is not so "sensitive" to bigger errors (does not square) and tells "how big of an error"
# we can expect from the forecast on average"
# ***
print('Prediction horizon = '+ str(lead_time),'Look back = ' + str(look_back))
trainScore = mean_absolute_error(trainY, trainPredict)
print('Train Score: %.2f MAE' % (trainScore))
testScore = mean_absolute_error(testY, testPredict)
print('Test Score: %.2f MAE' % (testScore))

'''# calculate root mean squared error. 
# weights "larger" errors more by squaring the values when calculating
print('Prediction horizon = '+ str(lead_time),'Look back = ' + str(look_back))
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))
'''
########################
######  write results to file
try:
    file_write = input("Save ouput file (y/n)? (Default = n)? ")
except ValueError:
    file_write = "n"

if file_write == "y":
    # make timestamp for unique filname
    stamp = str(time.clock())  #add timestamp for unique name
    stamp = stamp[0:2] 

    # generate filename and remove extra periods
    filename = 'FinErr_lstm_'+ str(n_epochs) + str(lead_time) + '_' + stamp + '.xlsx'    #example output file
    
    if filename.count('.') == 2:
        filename = filename.replace(".", "",1)
    writer = pd.ExcelWriter(filename)
    pd.DataFrame(trainPredict).to_excel(writer,'Train-predict') #save prediction output
    pd.DataFrame(trainY).to_excel(writer,'obs-train') #save observed output
    pd.DataFrame(testPredict).to_excel(writer,'Test-predict') #save output training data
    pd.DataFrame(testY).to_excel(writer,'obs_test') 
    writer.save()
    print('File saved in ', filename)

'''
# plot baseline and predictions
plt.close('all')
plt.plot(testY)
plt.plot(testPredict)
plt.show()
'''
