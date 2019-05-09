import numpy as np
import math
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error
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
def buildModel(outSize):
    regressor = Sequential()
    #regressor.add(Bidirectional(LSTM(units=50,return_sequences=True),input_shape = (X_train.shape[1], 1)))
    regressor.add(CuDNNLSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(CuDNNLSTM(units = 50,return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(CuDNNLSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = outSize))
    # Compiling
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return regressor


def getStationList():
    with open('pickles/stationList.pickle', 'rb') as handle:
        stationList = pickle.load(handle)
    os.chdir('excelFiles/LSTM')
    replaceDict = ['.xls','LSTMresult']
    for direct in glob.glob("*.xls"):
        fileName = direct                                               
        for w in replaceDict:
            fileName = fileName.replace(w,'')
        
        if fileName in stationList:
            stationList.remove(fileName)
            print(fileName)
    os.chdir('../..')
    result = stationList[38:57]
    print(result)
    return result
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

def fetchData(station,windosSize):

    with open('pickles/'+station+'2017trainRaw.pickle', 'rb') as handle:
        trainRawData = pickle.load(handle)
    with open('pickles/'+station+'2017testRaw.pickle', 'rb') as handle:
        testRawData = pickle.load(handle)
        
    sc, X_train, y_train, X_test, y_test = transfromData(trainRawData,testRawData,windosSize)
    return sc, X_train, y_train, X_test, y_test

def train(regressor,sc, X_train, y_train, X_test, y_test,epochs):
    for i in range(epochs):
        regressor.fit(X_train, y_train,validation_split=0.2, epochs = 1, batch_size = 32,verbose=2)
        predicted = sc.inverse_transform(regressor.predict(X_test))
        originY = sc.inverse_transform (y_test)
        mse = mean_squared_error(predicted, originY)
        print("Epoch : " +str(i)+", MSE : ["+str(mse)+"]")
        print('-------------------------------------------')
  
    return mse




epochs = 200
stationList = getStationList()
col=1
for station in stationList:
    print("training : " +station)
    MSEs = []
    for windowSize in range(7,31):
        sc, X_train, y_train, X_test, y_test = fetchData(station,windowSize)
        regressor = buildModel(windowSize-6)
        mse = train(regressor,sc,X_train, y_train, X_test, y_test,epochs)
        MSEs.append(mse)
    
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet1")
    writeExcelHead(sheet1,epochs,station)
    row = 1
    for m in MSEs:
        sheet1.write(row,1,m)
        row+=1
    book.save("excelFiles/LSTM/LSTMresult"+station+".xls")
        
    print('check point at ' + str(datetime.datetime.now()))

db.close()



