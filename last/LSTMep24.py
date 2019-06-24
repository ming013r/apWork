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

from tqdm import tqdm_notebook as tqdm


def mse(model,sc, X_train, y_train, X_test, y_test,epochs):
    MSEs = []
    for i in range(epochs):
        predicted = sc.inverse_transform(model.predict(X_test))
        originY = sc.inverse_transform (y_test)
        #mse = mean_squared_error(predicted, originY)
        for idx in len(predicted):
            mse = mean_squared_error(predicted[idx],originY[idx])
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


with open('ep24route.pickle', 'rb') as file:
    route_dict =pickle.load(file)
with open('stationList.pickle', 'rb') as handle:
    station_list = pickle.load(handle)
for station in tqdm(station_list):
    sc, X_train, y_train, X_test, y_test = fetchData(station,31)
    model = load_model(route_dict[station])
    
    MSEs = mse(model,sc,X_train, y_train, X_test, y_test,epochs)
 
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet1")
    
    for idx in len(MSEs):
        sheet1.write(row,1,MSEs[idx])
       
    book.save("excel/LSTMresult-"+station+".xls")

	