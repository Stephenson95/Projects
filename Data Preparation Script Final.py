# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:25:55 2020

@author: listep
"""
import pandas as pd
import os 

os.chdir(r'C:\Users\listep\Desktop\Education and Health portfolios performance audit')
HOR_data = pd.read_excel('Analysis of Government Responses to December 2019.xlsx', 
                         sheet_name = 'HoR outstanding responses',
                         skiprows = 21)

HOR_data = HOR_data.iloc[:,2:13]

HOR_data.columns = HOR_data.loc[0]

HOR_data.drop(index = 0, inplace = True)

HOR_data = HOR_data.loc[:, ['Report', 'Date of Government Response', 'Reporting period']]

HOR_data['Response By Day'] = pd.to_datetime(HOR_data['Date of Government Response'], format = '%Y/%m/%d', errors = 'coerce')

HOR_data['Response By Month'] = pd.to_datetime(HOR_data['Date of Government Response'], format = '%Y/%m/%d', errors = 'coerce').dt.to_period('m')

day_data = HOR_data['Response By Day'].value_counts().sort_index()

day_data = day_data.reindex(pd.date_range('2018-07-01', '2019-12-31')).fillna(0)

month_data = HOR_data['Response By Month'].value_counts().sort_index()

month_data = month_data.reindex(pd.period_range('2018-07-01', '2019-12-31', freq = 'M')).fillna(0)

reportperiod_data = HOR_data['Reporting period'].value_counts().sort_index(ascending = False)

#Export
day_data.to_csv('HOR_day_data.csv')

month_data.to_csv('HOR_month_data.csv')

reportperiod_data.to_csv('HOR_reportperiod_data.csv')

#Number of 'No response to date' and 'Partial response'
HOR_data['Date of Government Response'].value_counts()
#%%
import pandas as pd
import os 

Sen_data = pd.read_excel('Analysis of Government Responses to December 2019.xlsx', 
                         sheet_name = 'Senate outstanding responses',
                         skiprows = 18)

Sen_data = Sen_data.iloc[:,2:]

Sen_data.columns = Sen_data.loc[0]

Sen_data.drop(index = 0, inplace = True)

Sen_data = Sen_data.loc[:, ['Report', 'Date of Government Response', 'Reporting period']]

Sen_data['Response By Day'] = pd.to_datetime(Sen_data['Date of Government Response'], format = '%Y/%m/%d', errors = 'coerce')

Sen_data['Response By Month'] = pd.to_datetime(Sen_data['Date of Government Response'], format = '%Y/%m/%d', errors = 'coerce').dt.to_period('m')

day_data = Sen_data['Response By Day'].value_counts().sort_index()

day_data = day_data.reindex(pd.date_range('2018-01-01', '2019-09-30')).fillna(0)

month_data = Sen_data['Response By Month'].value_counts().sort_index()

month_data = month_data.reindex(pd.period_range('2018-01-01', '2019-09-30', freq = 'M')).fillna(0)

reportperiod_data = Sen_data['Reporting period'].value_counts().sort_index(ascending = False)

#Export
day_data.to_csv('Sen_day_data.csv')

month_data.to_csv('Sen_month_data.csv')

reportperiod_data.to_csv('Sen_reportperiod_data.csv')

Sen_data['Date of Government Response'].value_counts()
