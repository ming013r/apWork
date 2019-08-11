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
    start_dt = parse(year+'/01/01')
    ls = []
    for dt in daterange(start_dt, end_dt):
        ls.append(dt)
    return ls
def strHRKey(intHR):
    strHR = str(intHR)
    if intHR < 10 :
        strHR = '0'+strHR
    else:
        strHR = int(strHR)
    return strHR


def resembleExcel_asDict(fname):
    df = pd.read_excel(fname)
    station = ''
    year = ''
    main_dict = {}
    for row_idx in range(len(df)):
        row = df.iloc[row_idx]
        station = row['測站']
        year = str(parse(row['日期']).year)
        if row['測項'] == 'PM2.5':
            for i in range(3,27):
                datetime = parse(row['日期'])+timedelta(hours=int(i-3))
                val = str(row[strHRKey(i-3)])
                processed_val = 0.0
                try:
                    processed_val = float(val)
                    if math.isnan(processed_val):
                        processed_val = 'NULL'
                except ValueError:
                    processed_val = 'NULL'
                main_dict[datetime] = processed_val
    return main_dict, station, year


## Main program
for dir in tqdm(glob.glob('raw/*')):
    for file in glob.glob(dir+'/*.xls'):
        raw_dict, station, year = resembleExcel_asDict(file)
        day_list = get_all_days(year)
        for day in day_list:
            hr_list = hrs_in_a_day(day)
            for hr in hr_list:
                try:
                    hasVal = raw_dict[hr]
                except KeyError:
                    raw_dict[hr] = 'NULL'
        print(station+year,len(raw_dict))
        with open('data_src/'+station+year+'full_dict.pickle', 'wb') as handle:
            pickle.dump(raw_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


###deserialization
with open('stationList.pickle', 'rb') as handle:
    stationList = pickle.load(handle)
#二林2017full_dict.pickle
for station in stationList:
    trainData_of_a_station = []
    year_days = get_all_days('2016')
    with open('data_src/dict/'+station+'2016full_dict.pickle', 'rb') as handle:
        dict_of_a_station = pickle.load(handle)
    for day in year_days:
        trainData_of_a_station.append(dict_of_a_station[day])
        year_days = get_all_days('2017')
    with open('data_src/dict/'+station+'2017full_dict.pickle', 'rb') as handle:
        dict_of_a_station = pickle.load(handle)
    for day in year_days:
        trainData_of_a_station.append(dict_of_a_station[day])
    with open('data_src/array/'+station+year+'train.pickle', 'wb') as handle:
        pickle.dump(raw_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
