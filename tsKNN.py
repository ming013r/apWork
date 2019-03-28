import numpy as np
import math
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error
import os, glob
import pickle

trainData = pickle.load( open( "data.p", "rb" ) )
for i in range(77):
    trainData[i]=trainData[i].reshape(-1,1)

from tslearn.generators import random_walk_blobs
from tslearn.preprocessing import TimeSeriesScalerMinMax, to_time_series_dataset
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeries
from tslearn.piecewise import SymbolicAggregateApproximation


time_series = to_time_series_dataset(trainData)
knn = KNeighborsTimeSeries(n_neighbors=1).fit(time_series)


ind = knn.kneighbors(to_time_series_dataset(trainData), return_distance=False)
print(ind)
