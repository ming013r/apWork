import numpy as np
import math
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd


import sys
import xlwt, xlrd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import datetime,pickle,os,glob

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional,CuDNNLSTM 
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

import sys
import xlwt
from sklearn.preprocessing import MinMaxScaler
import datetime,pickle,os,glob

def mse(model,sc, X_train, y_train, X_test, y_test):
    MSEs = []
    predicted = sc.inverse_transform(model.predict(X_test))
    originY = sc.inverse_transform (y_test)
        #mse = mean_squared_error(predicted, originY)
    for idx in range(24):
        mse = mean_squared_error(predicted[:,idx],originY[:,idx])
        MSEs.append(mse)
    return MSEs
def fetchData(station,windosSize):
    with open('pickles/'+station+'2017trainRaw.pickle', 'rb') as handle:
        trainRawData = pickle.load(handle)
    with open('pickles/'+station+'2017testRaw.pickle', 'rb') as handle:
        testRawData = pickle.load(handle)
        
    sc, X_train, y_train, X_test, y_test = transfromData(trainRawData,testRawData,windosSize)
    return sc, X_train, y_train, X_test, y_test

def transfromData(trainRaw, testRaw,windosSize):  ##Train ratial, train, test
    sc = MinMaxScaler(feature_range = (0, 1))

    npTrain = sc.fit_transform(np.array(trainRaw).reshape(-1,1))
    npTest = sc.fit_transform(np.array(testRaw).reshape(-1,1))
    
    X_train, y_train = splitXy(npTrain,windosSize)
    X_test, y_test = splitXy(npTest,windosSize)
    return sc, X_train, y_train, X_test, y_test

def splitXy(data,windosSize):
    windows = []
    for i in range(windosSize, data.shape[0]):
        windows.append(data[i-windosSize:i, 0])
    np.random.shuffle(windows)
    X = []
    y = []
    for i in range(len(windows)):
        X.append(windows[i][:6])
        y.append(windows[i][-(windosSize-6):])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X,y
def buildModel():
    regressor = Sequential()
    #regressor.add(Bidirectional(LSTM(units=50,return_sequences=True),input_shape = (X_train.shape[1], 1)))
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50,return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 24))
    # Compiling
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return regressor

with open('ep24route.pickle', 'rb') as file:
    route_dict =pickle.load(file)
with open('stationList.pickle', 'rb') as handle:
    station_list = pickle.load(handle)
for station in tqdm(station_list):
    sc, X_train, y_train, X_test, y_test = fetchData(station,30)
    model = buildModel()
    model = load_model(route_dict[station])
    
    MSEs = mse(model,sc,X_train, y_train, X_test, y_test)
 
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet1")
    
    for idx in range(len(MSEs)):
        sheet1.write(idx,1,MSEs[idx])
       
    book.save("excel2/LSTMresult-"+station+".xls")

	