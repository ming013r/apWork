import pandas as pd # 引用套件並縮寫為 pd
import pickle
import math
import numpy as np
from tqdm import tqdm
import time
import operator
from datetime import timedelta, date
from dateutil.parser import parse
import datetime
import glob
import hashlib

#Functions
def hrs_in_a_day(dt):
    hr_list = []
    for i in range(24):
        day_dt = dt + timedelta(hours=i)
        hr_list.append(day_dt)
    return hr_list
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)
def get_all_days(year):
    end_dt = parse(year+'/12/31')
    start_dt = parse(year+'/09/01')
    ls = []
    for dt in daterange(start_dt, end_dt):
        ls.append(dt)
    return ls



###deserialization
with open('stationList.pickle', 'rb') as handle:
    stationList = pickle.load(handle)
#二林2017full_dict.pickle
for station in stationList:
    trainData_of_a_station = []
    #write 2016
    year_days = get_all_days('2016')
    with open('data_src/dict/'+station+'2016full_dict.pickle', 'rb') as handle:
        dict_of_a_station = pickle.load(handle)
    for day in year_days:
        trainData_of_a_station.append(dict_of_a_station[day])
    #write 2017
    year_days = get_all_days('2017')
    with open('data_src/dict/'+station+'2017full_dict.pickle', 'rb') as handle:
        dict_of_a_station = pickle.load(handle)
    for day in year_days:
        trainData_of_a_station.append(dict_of_a_station[day])
    print(len(trainData_of_a_station))
    with open('data_src/array/'+station+'train.pickle', 'wb') as handle:
        pickle.dump(trainData_of_a_station, handle, protocol=pickle.HIGHEST_PROTOCOL)


    testData_of_a_station = []
    year_days = get_all_days('2018')
    with open('data_src/dict/'+station+'2018full_dict.pickle', 'rb') as handle:
        dict_of_a_station = pickle.load(handle)
    for day in year_days:
        testData_of_a_station.append(dict_of_a_station[day])
    print(len(testData_of_a_station))
    with open('data_src/array/'+station+'test.pickle', 'wb') as handle:
        pickle.dump(testData_of_a_station, handle, protocol=pickle.HIGHEST_PROTOCOL)
