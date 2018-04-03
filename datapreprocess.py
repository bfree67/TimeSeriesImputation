# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 23:46:11 2018

@author: Brian
"""
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler #, StandardScaler
import time
import sys
from pandas import read_csv

np.random.seed(7)

def countcensor(dfx):
    names = list(dfx)
    censor_count = []
    dfx_length = len(dfx.columns)
    print("\nCensored data in each feature:")
    for i in range(dfx_length):
        censor_count.append((dfx.iloc[:,i]<=0.).sum())  ## count values and add to list
        print('{0:4s} {1:2d}'.format(names[i], censor_count[i]))
   
    try:
        censor_correct = input("Convert censored and negative data to lowest positive value? (Default = n)? ")
    except ValueError:
        censor_correct = "y"
        
    if censor_correct == "y":
        ##### if feature has censored data or negative data, replace it 
        ##### with the smallest positive, non-zero value
        print("\nFeature minimum positive values:") ### for space
        
        for i in range(dfx_length):
            if censor_count[i]>0:
                xarray = np.asarray(dfx.iloc[:,i].sort_values())  #sort column and convert to array
                xmin = xarray[censor_count[i]+1] #select smallest, positive, non-zero value
                dfx.iloc[:,i][dfx.iloc[:,i]<= 0.] = xmin  ### replace censored values with min value
            else:
                xmin - 0.
            print('{0:4s} {1:3f}'.format(names[i], xmin))        
        print("\nCensored and negative values converted to feature's smallest positive, non-zero value.")
        
    else:
        print("\nCensored and negative values not converted.")
 
    return dfx
#################################################################

def checknans(dfx):
    names = list(dfx)
    nan_count = []
    dfx_length = len(dfx.columns)
    print("\nNaN data in each feature:")
    for i in range(dfx_length):
        nan_count.append(dfx.iloc[:,i].isnull().sum())  ## count values and add to list
        print('{0:4s} {1:2d}'.format(names[i], nan_count[i]))
    
    
    return dfx

def preprocess(x):
    return x+2
# 
# ***
# 1) load dataset
# ***
data_file_name = 'air_dataset.csv'
df = read_csv(data_file_name, engine='python', skipfooter=3)

dft = df[['O3','SO2','NO2', 'WS','TEMP','ATP', 'SR']]

a = list(dft)  #makes a list of the column names in the dataframe

for i in range (len(a)):    #prints a list of the column names
    print (i, a[i])

last_col = np.shape(dft)[1] - 1
# pick column to predict
try:
    target_col = int(input("Select the column number to predict (default = " + a[last_col] + "): "))
except ValueError:
    target_col = last_col
    
npt = np.asmatrix(dft) ### convert to np matrix

df_cleansed1 = countcensor(dft)
df_nan = checknans(df_cleansed1)


