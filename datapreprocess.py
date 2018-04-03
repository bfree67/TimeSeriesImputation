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
    '''
    function to identify and count values in dataframe columns that are censored (==0) or
    negative values. Asks to convert the values to the lowest positive, non-zero value in the 
    column.
    Input = dataframe of raw data
    Output = dataframe of data cleaned of censored values or original data
    '''    
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
                xmin = 0.
            print('{0:4s} {1:3f}'.format(names[i], xmin))        
        print("\nCensored and negative values converted to feature's smallest positive, non-zero value.")
        
    else:
        print("\nCensored and negative values not converted.")
 
    return dfx
#################################################################

def checkoutliers(dfx):
    '''
    function to identify and count possible outliers in dataframe columns 
    Input = dataframe of data
    Output = none
    ''' 
    names = list(dfx)
    out_count = []
    dfx_length = len(dfx.columns)
    print("\nPossible outliers in each feature:")
    for i in range(dfx_length):
        dfx_test = 3*dfx.iloc[:,i].mean()  ##define test for outlier
        out_count.append((dfx.iloc[:,i]>dfx_test).sum())  ## count values and add to list
        print('{0:4s} {1:2d}'.format(names[i], out_count[i]))
    return

def checknans(dfx):
    '''
    function to identify and count NaNs in dataframe columns 
    Input = dataframe of data
    Output = none
    ''' 
    names = list(dfx)
    nan_count = []
    dfx_length = len(dfx.columns)
    print("\nNaN data in each feature:")
    for i in range(dfx_length):
        nan_count.append(dfx.iloc[:,i].isnull().sum())  ## count values and add to list
        print('{0:4s} {1:2d}'.format(names[i], nan_count[i]))
    return

def badzeros(dfx):
    '''
    function to convert NaN values to 0 and add a column that shows them as NaNs
    Input = dataframe of data and target column to replace NaNs with zeros
    Output = numpy matrix of data concatenated with binary filters and dataframe of
    features and first occurance of NaN and outlier.
    ''' 
    ###### call other local functions to replace censored data and 
    ###### check for NaNs and outliers
    dfx = countcensor(dfx)
    checknans(dfx)
    checkoutliers(dfx)
    
    names = list(dfx)
    
    np_nan = np.matrix(dfx)  #convert dataframe to matrix
    
    np_clean =  np.nan_to_num(np_nan) #convert NaNs to 0's
    
    np_row,np_col = np.shape(np_nan)
    np_z1 = np.zeros((np_row,np_col))  #make a binary filter for where NaNs were
    np_z1 = (np_clean == 0.) + 0.
    
    np_z2 = np.zeros((np_row,np_col)) #make a binary filter for possible outliers
    np_z2 = (np_clean > np_clean.mean(axis=0)*3) + 0.
    
    np_ready = np.concatenate((np_clean, np_z1, np_z2), axis=1)  #merge matrices together
    
    a = []
    for i in range(np_col):
        arnan = np_ready[:,i+np_col]
        arout = np_ready[:,i+(2*np_col)]
        
        tnan = np.nonzero(arnan == 1.)
        if arnan.sum()>0.:
            tnan_first = tnan[0][0]
        else:
            tnan_first = np_row
        
        tout = np.nonzero(arout == 1.)
        if arout.sum()>0.:
            tout_first = tout[0][0]
        else:
            tout_first = np_row   
        
        a.append(names[i])
        a.append(tnan_first)
        a.append(tout_first)
        
        col_names = ["Name","1st NaN", "1st Outlier"]
        dfa = pd.DataFrame(np.array(a).reshape(-1,3), columns = col_names)
    
    return np_ready, dfa
# 
# ***
# 1) load dataset
# ***
'''
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
    
#npt = np.asmatrix(dft) ### convert to np matrix

#df_cleansed1 = countcensor(dft)
#checknans(df_cleansed1)
#checkoutliers(df_cleansed1)
#np_dft, a_dft = badzeros(dft)
'''



