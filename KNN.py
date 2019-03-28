import numpy as np
import math
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error
import os, glob

def getStationList():
    stationList = []
    os.chdir("C:/Users/ming/Desktop/Workspace/AirPolution")
    for direct in glob.glob("*"):
        if os.path.isdir(direct):
            os.chdir(direct)
            for file in glob.glob("*.xls"):
                 if "106" in file:
                    stationList.append(file.strip("106年站_20180309.xls"))
            os.chdir("..")  
    return stationList

windosSize = 20
def transfromData(rawData): 
    sc = MinMaxScaler(feature_range = (0, 1))
    npRaw = np.array(rawData)
    scaledData = sc.fit_transform(npRaw.reshape(-1,1))
    return scaledData
                                  
import pickle
trainData = pickle.load( open( "data.p", "rb" ) )


from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=3)
neigh.fit(trainData)
result = neigh.kneighbors(trainData, return_distance=False)

