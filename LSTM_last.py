import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

import sys
import xlwt
from sklearn.preprocessing import MinMaxScaler
import datetime,pickle,os,glob

from tqdm import tqdm

def getStationList():
    with open('pickles/stationList.pickle', 'rb') as handle:
        stationList = pickle.load(handle)
    os.chdir('excelFiles/LSTM')
    replaceDict = ['.xls','LSTMresult-']
    for direct in glob.glob("*.xls"):
        fileName = direct
        for w in replaceDict:
            fileName = fileName.replace(w,'')

        if fileName in stationList:
            stationList.remove(fileName)
            print(fileName)
    os.chdir('../..')
    return stationList
def clean_NULL_values(data, windowSize):
    windows = []
    for i in range(6, len(data)-windowSize):
        a_data = data[i-6:i]
        a_data.append(data[i+windowSize-7])
        if 'NULL' not in a_data:
            windows.append(a_data)
    sc = MinMaxScaler(feature_range = (0, 1))
    npdata = sc.fit_transform(np.array(windows).reshape(-1,1))
    return npdata, sc

def transfromData(trainRaw, testRaw,windowSize):  ##Train ratial, train, test
    X_test, y_test,sc = splitXy(testRaw,windowSize)
    X_train, y_train, sc = splitXy(trainRaw,windowSize)
    return sc, X_train, y_train, X_test, y_test

def splitXy(rawdata, windowSize):
    data, sc = clean_NULL_values(rawdata,windowSize)
    windows = []
    for i in range(6, data.shape[0]-windowSize):
        a_data = data[i-6:i, 0].tolist()
        a_data.append(data[i])
        windows.append(a_data)
    np.random.shuffle(windows)
    X = []
    y = []
    for i in range(len(windows)):
        X.append(windows[i][:6])
        y.append(windows[i][-1:][0])
    X, y = np.array(X), np.array(y)
    print(y.shape)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X,y, sc
def buildModel():
    regressor = Sequential()
    #regressor.add(Bidirectional(LSTM(units=50,return_sequences=True),input_shape = (X_train.shape[1], 1)))
    regressor.add(LSTM (units = 50, return_sequences = True, input_shape = (6, 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM (units = 50,return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM (units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    # Compiling
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return regressor


def Visualize():
    predicted = sc.inverse_transform(regressor.predict(X_test))
    originY = sc.inverse_transform (y_test)
    print("MSE : ["+str(mean_squared_error(predicted, originY))+"]")
    # Visualising the results
    plt.plot(originY[:100], color = 'red', label = 'Real')
    plt.plot(predicted[:100], color = 'blue', label = 'Predicted ')
    plt.legend()
    plt.show()
def writeExcelHead(sheet1,epochs,station):
    sheet1.write(0,1,station)
    raw = 1
    for e in range(epochs):
        sheet1.write(raw,0,e+1)
        raw+=1

##get data##
def fetchData(station,windowSize):

    with open('data_src/array/'+station+'train.pickle', 'rb') as handle:
        trainRawData = pickle.load(handle)
    with open('data_src/array/'+station+'train.pickle', 'rb') as handle:
        testRawData = pickle.load(handle)

    sc, X_train, y_train, X_test, y_test = transfromData(trainRawData,testRawData,windowSize)
    return sc, X_train, y_train, X_test, y_test

def train(model,epochs,windowSize,station):

    sc, X_train, y_train, X_test, y_test = fetchData(station,windowSize)
    for i in range(epochs):
        model.fit(X_train, y_train,validation_split=0.2, epochs = 1, batch_size = 32,verbose=0)
        #print('Current Epoch:',i,'time steps : ', windowSize-6)


    predicted = sc.inverse_transform(model.predict(X_test))
    originY = sc.inverse_transform (y_test)

    mse = mean_squared_error(predicted, originY)
    mae = mean_absolute_error(predicted,originY)

    model.save('model/LSTM/LSTM'+station+str(windowSize-6)+'.h5')
    return mse,mae


epochs = 250
stationList = getStationList()
col=1
for station in stationList:
    print("training : " +station)
    MSEs = []
    MAEs = []
    for windowSize in tqdm(range(7,31)):
        model = buildModel()
        mse,mae = train(model,epochs,windowSize,station)
        MSEs.append(mse)
        MAEs.append(mae)

    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet1")
    writeExcelHead(sheet1,epochs,station)
    row = 1
    for m in MSEs:
        sheet1.write(row,1,m)
        row+=1
    row = 1
    for m in MAEs:
        sheet1.write(row,2,m)
        row+=1
    book.save("excelFiles/LSTM/LSTMresult-"+station+".xls")

    print('check point at ' + str(datetime.datetime.now()))
