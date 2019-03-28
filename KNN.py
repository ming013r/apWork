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
                                  
								  
with open('Output.txt', 'r') as myfile:
  data = myfile.read()
import json
trainData = json.loads(data)




from tslearn.generators import random_walk_blobs
from tslearn.preprocessing import TimeSeriesScalerMinMax, to_time_series_dataset
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeries
from tslearn.piecewise import SymbolicAggregateApproximation
import numpy as np

time_series = to_time_series_dataset(trainData)
knn = KNeighborsTimeSeries(n_neighbors=1).fit(time_series)


ind = knn.kneighbors(to_time_series_dataset(trainData), return_distance=False)
print(ind)




