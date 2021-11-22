# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:14:03 2020

@author: listep
"""
import pandas as pd
import numpy as np
import os

#Similary Strings
import difflib

def similarity_match(data, col, threshold):
    
    global counter
    
    data.reset_index(drop = True, inplace = True)
    
    counter = 0
    
    data['Similarity_Match'] = None
    
    for index, i in enumerate(data[col]):
        
        if not(pd.isna(data.loc[index, 'Similarity_Match'])):
            
            continue
            
        else:
            
            if len(difflib.get_close_matches(i, data[col], cutoff = threshold)) > 1:
                
                data.loc[data[col].isin(difflib.get_close_matches(i, data[col], cutoff = threshold)), 'Similarity_Match'] = counter
                
                counter += 1
            
            else:
                
                data.loc[index, 'Similarity_Match'] = counter
                
                counter += 1
            
            
def analysis_output(data):
    #Preparation
    period_list = data['Reporting period'].unique().tolist()
    
    period_list.sort()
    
    reporting_period = period_list[-2:] #2 latest reporting periods required
    
    analysis_df = data[data['Reporting period'].isin(reporting_period)]
    
    for col in ['Parliamentary Committee', 'Committee Type', 'Report']:
        
        analysis_df[col] = analysis_df[col].apply(lambda i : i.strip().title())
    
    analysis_df['Unique ID'] = analysis_df['Parliamentary Committee'] + analysis_df['Committee Type'] + analysis_df['Report'] #Create a unique identifier
    
    #Apply string match (fuzzy matching)
    similarity_match(analysis_df,'Unique ID', 0.95)
    
    #No. of reports with response
    reports_with_response = analysis_df[analysis_df['Date of Government Response'] != 'No response to date']['Similarity_Match'].nunique()
    
    #No. of reports with a response received within timeframe
    subset1 = analysis_df['Date of Government Response'] != 'No response to date' #Response Received
    
    subset2 = (analysis_df['Responded within specified time period?'] == 'Yes')
    
    reports_within_timeframe = analysis_df[np.multiply(subset1, subset2)]['Similarity_Match'].nunique()
    
    #No. of reports response received late
    subset1 = analysis_df['Date of Government Response'] != 'No response to date'
    
    subset2 = (analysis_df['Responded within specified time period?'] == 'No')
    
    reports_not_within_timeframe = analysis_df[np.multiply(subset1, subset2)]['Similarity_Match'].nunique()
    
    #No. of reports with no response
    list_reports_responded = analysis_df[analysis_df['Date of Government Response'] != 'No response to date']['Similarity_Match'].unique().tolist() #Remove reports that have been responded to at a later period
    
    list_reports_not_responded = analysis_df[analysis_df['Date of Government Response'] == 'No response to date']['Similarity_Match'].unique().tolist()
    
    report_with_no_response = len(list(set(list_reports_not_responded) - set(list_reports_responded)))
    
    #Total number of reports in schedule
    number_of_reports = analysis_df['Similarity_Match'].nunique()
    
    #Shortest timeframe taken
    try:
        shortest_time = min(analysis_df['Time taken to respond in months'][~pd.isna(analysis_df['Time taken to respond in months'])])
        
    except:
        
        analysis_df['Time taken to respond in months'].replace('?', np.nan, inplace = True)
        
        shortest_time = min(analysis_df['Time taken to respond in months'][~pd.isna(analysis_df['Time taken to respond in months'])])
        
    
    #Longest response time where response was provided
    longest_time = max(analysis_df['Time taken to respond in months'][~pd.isna(analysis_df['Time taken to respond in months'])])
    
    #Latest pending response not yet received
    latest_pending_response = max(analysis_df['Lateness in months where government response not yet provided'])
    
    #Output to dataframe
    output = pd.DataFrame(data = {'No. of reports with response':reports_with_response, 'No. of reports with a response that were received within the specified timeframe':reports_within_timeframe,
                           'No of reports with a response but received late':reports_not_within_timeframe, 'No. of reports with no response':report_with_no_response,
                           'Total number of reports included in the schedule':number_of_reports, 'Shortest timeframe taken to respond (months)':shortest_time, 
                           'Longest response time where a response was provided (months)':longest_time, 'Latest pending response (not yet received) (months)':latest_pending_response}, index = [0])
    
    output = output.T
    
    output.reset_index(inplace = True)
    
    output.columns = ['Description', 'Amount']
    
    output['Percentage (%)'] = np.nan
    
    output.loc[0:4, 'Percentage (%)'] = round(output['Amount']*100/number_of_reports, 0)
    
    return output


HOR_output = analysis_output(final_hor_data)

SEN_output = analysis_output(final_sen_data)

#%%
#Preparing Data for Visualisations
def visualisation_prep(data):
    
    timeline_data = data.loc[:, ['Report', 'Date of Government Response', 'Reporting period']]
    
    timeline_data['Response By Day'] = pd.to_datetime(timeline_data['Date of Government Response'], format = '%Y/%m/%d', errors = 'coerce')
    
    timeline_data['Response By Month'] = pd.to_datetime(timeline_data['Date of Government Response'], format = '%Y/%m/%d', errors = 'coerce').dt.to_period('m')
    
    day_data = timeline_data['Response By Day'].value_counts().sort_index()
    
    day_data = day_data.reindex(pd.date_range('2018-07-01', '2020-06-30')).fillna(0).reset_index()
    
    month_data = timeline_data['Response By Month'].value_counts().sort_index()
    
    month_data = month_data.reindex(pd.period_range('2018-07-01', '2020-06-30', freq = 'M')).fillna(0).reset_index()
    
    month_data['index'] = month_data['index'].astype(str)
    
    month_data['index']= pd.to_datetime(month_data['index'])
    
    reportperiod_data = timeline_data['Reporting period'].value_counts().sort_index(ascending = False).reset_index()

    return day_data, month_data, reportperiod_data

hor_day, hor_month, hor_reportperiod = visualisation_prep(final_hor_data)

sen_day, sen_month, sen_reportpreiod = visualisation_prep(final_sen_data)
#%%
#Create Visualisation
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.ticker as mtick
from datetime import datetime

def create_visualisations(day_data, month_data, reportperiod_data, title):
    
    latest_response_date = max(reportperiod_data['index']).strftime('%d-%m-%Y')
    
    #Generate Visualisations
    plt.rc('font', size=12)
    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False, figsize=(30, 16))
    
    #Specify values for lines in the timeseries
    ax[0].plot(day_data['index'], day_data['Response By Day'],\
            color = 'mediumblue', label = 'No. of Responses')
    ax[1].plot(month_data['index'], month_data['Response By Month'],\
            color = 'cornflowerblue', label = 'No. of Responses')
    
    #Set the x-axis major ticks
    for i in [0,1]:
        ax[i].xaxis.set_major_locator(MonthLocator())
        ax[i].xaxis.set_major_formatter(DateFormatter('%b-%y'))
    
    #Set the y-axis major ticks
    ax[0].yaxis.set_major_locator(mtick.MultipleLocator(1))
    
    ax[1].yaxis.set_major_locator(mtick.MultipleLocator(1))
    ax[1].yaxis.set_minor_locator(mtick.MultipleLocator(1))    
    
#    #Set x-axis limit
#    for i in [0,1]:
#        ax[i].set_xlim([datetime(2018,7,1), datetime(2020,7,31)])
#    
    #Set reference line for COVID-19 Government Pandemic Response date:
    for i in [0,1]:
        ax[i].axvline(x = datetime(2019,8,7), color = 'red', label = 'PMC Secretary\'s Announcement to Respond on Time', ls = '--')
    
    #Time series formatting
    ax[0].set_xlabel('Response Date')
    ax[0].set_ylabel('No. of Government Response')
    ax[0].set_title('Daily Time Series of {} Government Response to {}'.format(title, latest_response_date))
    ax[0].grid(True)
    ax[0].legend(loc = 'upper left')
    
    ax[1].set_xlabel('Response Date')
    ax[1].set_ylabel('No. of Government Response')
    ax[1].set_title('Monthly Time Series of {} Government Response to {}'.format(title, latest_response_date))
    ax[1].grid(True)
    ax[1].legend(loc = 'upper left')
    
    plt.savefig('{} time series'.format(title))
    
    
create_visualisations(hor_day, hor_month, hor_reportperiod, 'House of Reps')


create_visualisations(sen_day, hor_month, hor_reportperiod, 'Senate')

#%%
#Output Files
import tkinter as tk
from tkinter.filedialog import askdirectory

root = tk.Tk()
root.withdraw()
os.chdir(askdirectory(title = 'Selecting output directory'))


with pd.ExcelWriter('Sen and Hor Response Statistics.xlsx') as writer:
    HOR_output.to_excel(writer, sheet_name = 'HOR Descriptive Table', index = False)
    SEN_output.to_excel(writer, sheet_name = 'SEN Descriptive Table', index = False)


#%%
#Additional Footnotes
def get_recent_two_periods(data):
    period_list = data['Reporting period'].unique().tolist()
        
    period_list.sort()
    
    reporting_period = period_list[-2:] #2 latest reporting periods required
    
    analysis_df = data[data['Reporting period'].isin(reporting_period)]
    
    for col in ['Parliamentary Committee', 'Committee Type', 'Report']:
        
        analysis_df[col] = analysis_df[col].apply(lambda i : i.strip().title())
    
    analysis_df['Unique ID'] = analysis_df['Parliamentary Committee'] + analysis_df['Committee Type'] + analysis_df['Report'] #Create a unique identifier
    
    #Apply string match (fuzzy matching)
    similarity_match(analysis_df,'Unique ID', 0.95)
    
    return analysis_df

def additional_footnotes(data, partial_response_field, yet_to_expire_field):
    #Number of partial responses
    print('Number of Partial Responses: ', data[data['Date of Government Response'] == partial_response_field]['Similarity_Match'].nunique())
    
    #Look at number of reports yet to expire
    list_reports_once_not_expired = data[data['Responded within specified time period?'].isin(yet_to_expire_field)]['Similarity_Match'].unique().tolist()
    
    list_reports_now_expired = data[~data['Responded within specified time period?'].isin(yet_to_expire_field)]['Similarity_Match'].unique().tolist()
    
    print('Number of Reports yet to expire: ', len(list(set(list_reports_once_not_expired) - set(list_reports_now_expired))))
    
    #Look at number of reports with no response
    list_reports_responded = data[data['Date of Government Response'] != 'No response to date']['Similarity_Match'].unique().tolist() #Remove reports that have been responded to at a later period
        
    list_reports_not_responded = data[data['Date of Government Response'] == 'No response to date']['Similarity_Match'].unique().tolist()
    
    print('Number of reports with no response: ', len(list(set(list_reports_not_responded) - set(list_reports_responded))))
    
#####################################################################
#Senate Data      
sen_analysis_df = get_recent_two_periods(final_sen_data)

additional_footnotes(sen_analysis_df,'Partial government response provided during debate of the bill', ['Time not expired'])

#JCPAA (Public Accounts and Audit) #Established by Public Accounts and Audit
sen_JCPAA = sen_analysis_df[sen_analysis_df['Parliamentary Committee'] == 'Public Accounts And Audit']

sen_JCPAA['Similarity_Match'].nunique()

sen_JCPAA.sort_values(by = 'Reporting period', ascending = False, inplace = True)

test = sen_JCPAA.drop_duplicates(['Similarity_Match'], keep = 'first') #Issue identified with data quality

test['Date of Government Response'].value_counts(dropna = False)
#####################################################################
#HOR data
hor_analysis_df = get_recent_two_periods(final_hor_data)

additional_footnotes(hor_analysis_df, 'Partial response', ['Time not expired', 'Time has not expired'])

#JCPAA (Public Accounts and Audit) #Established by Public Accounts and Audit
hor_JCPAA = hor_analysis_df[hor_analysis_df['Parliamentary Committee'] == 'Public Accounts And Audit']

hor_JCPAA['Similarity_Match'].nunique()

hor_JCPAA.sort_values(by = 'Reporting period', ascending = False, inplace = True)

test = hor_JCPAA.drop_duplicates(['Similarity_Match'], keep = 'first') #Issue identified with data quality

test['Date of Government Response'].value_counts(dropna = False)

