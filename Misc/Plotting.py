# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:07:54 2020

@author: khorso
"""

#%%
#Import in standard modules that need to be used (expand as needed)
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.ticker as mtick
from datetime import datetime

#Ignore depreciated warnings
warnings.filterwarnings("ignore")

#%%
#UI
import tkinter as tk
from tkinter.simpledialog import askstring
from tkinter.filedialog import askopenfilename, askdirectory

#Designating input directory
root = tk.Tk()

def input_directory():
    os.chdir(askdirectory(title = 'Selecting input directory'))

input_directory()

#Creating output directory
outputdirectory = askstring(title = 'Designate an output directory', \
                            prompt = 'Please title the output directory: ')
time = datetime.now().strftime("%a %d.%b.%Y")
try:
    os.makedirs(r".\\"+outputdirectory+'-'+time, exist_ok=True)
except:
    print("A folder for ' " + time + " ' already exists."+\
          " No extra folder created")

#file_select function for selecting files and relevant_sheet
def file_select():    
    infile = askopenfilename(title='Select a data file')

    if infile.endswith('.xlsx'):
        sheetname = askstring(title = 'Specify the sheet name', \
                              prompt = 'Please specify the sheet name. If there is only one sheet,'+\
                              ' enter 0 as the input: ')
    
        if sheetname == '0':
            sheetname = int(sheetname)
        
        data = pd.read_excel(infile, sheet_name = sheetname)
    
    elif infile.endswith('.csv'):
        data = pd.read_csv(infile)
        
    else:
        print ('File format is not in .xlsx or .csv')

    return data

#%%
#Read in files to be used
austenderdf = file_select()
#SONdf = file_select()

#Assertion check for uniqueness of Contract IDs i.e no duplicates in dataset
assert austenderdf['Contract ID'].nunique() == len(austenderdf)

#Filter for limited tender contracts in austender dataset
limitedtenderdf = austenderdf[austenderdf['Procurement Method'] == 'Limited tender']

#Filter for limited tenders > $80,000 in austender dataset
limitedtenderdfvalue = limitedtenderdf[limitedtenderdf['Applicable Value'] > 80000 ]

#Filter limited tenders from the 2019-20 FY
limitedtenderdf1920 = limitedtenderdfvalue[limitedtenderdfvalue['Applicable FY Publish Date'] == \
                                           '2019-2020']

#Groupby and sort for the top ten agencies and total amount spent
limitedtenderdfamt= limitedtenderdf1920.groupby(['Agency Name']).agg({'Applicable Value': 'sum',\
                                                                     'Contract ID': 'count'})
limitedtenderdfamt = limitedtenderdfamt.reset_index().rename(columns = {'Contract ID': 'No. of Contracts'})

top10limitedtenderamt = limitedtenderdfamt.nlargest(10, columns = 'Applicable Value')

#Filter for Department of Health
healthdf = limitedtenderdf1920[limitedtenderdf1920['Agency Name'] == 'Department of Health']

#Clean up descriptions in the austender dataset
for i in ['\n', '</tr>', '</td>', '<tbody>', '</tbody>', '</table>', '<tr>', '</th>', '&nbsp;',
          '<td>', '<th>', '<table class="medium-th">', '<span>', '</span>', '<p>', '</p>',
          '<table width="141" border="0" cellspacing="0" cellpadding="0">',
          '<table width="172" border="0" cellspacing="0" cellpadding="0">',
          '<table width="356" border="0" cellspacing="0" cellpadding="0">',
          '<table width="366" border="0" cellspacing="0" cellpadding="0">',
          '<td class="xl64" width="356" height="21">','<td class="xl64" width="356" height="21">',
          '<td class="xl68" width="141" height="20">','<td class="xl68" width="141" height="21">',
          '<td class="xl69" width="141" height="20">','<td class="xl66" width="141" height="20">',
          '<td class="xl66" width="141" height="21">','<td class="xl66" width="356" height="20">', 
          '<td class="xl66" width="356" height="21">', '<td class="xl66" width="172" height="21">',
          '<td class="xl66" width="172" height="20">', '<td class="xl64" width="366" height="20">',
          '<td id="WD31" class="xl64" width="172" height="21">']:
    healthdf['Description'] = healthdf['Description'].apply(lambda x: x.replace(i, ''))

healthdf['Description'] = healthdf['Description'].apply(lambda x: x.strip())

#Groupby and sort for the top ten descriptions:
healthdfdesc = healthdf.groupby(['Agency Name', 'Description']).agg({'Applicable Value': 'sum',\
                                                                     'Contract ID': 'count'})
healthdfdesc = healthdfdesc.reset_index().rename(columns = {'Contract ID': 'No. of Contracts'})

top10healthdesc = healthdfdesc.nlargest(10, columns = 'Applicable Value')

#Groupby and sort for the top ten categories:
healthdfcat = healthdf.groupby(['Agency Name', 'UNSPSC Title']).agg({'Applicable Value': 'sum',\
                                                                     'Contract ID': 'count'})
healthdfcat = healthdfcat.reset_index().rename(columns = {'Contract ID': 'No. of Contracts'})

top10healthcat = healthdfcat.nlargest(10, columns = 'Applicable Value')

#Groupby and sort for the top ten suppliers:
healthdfsupplier = healthdf.groupby(['Agency Name', 'Supplier Name']).agg({'Applicable Value': 'sum',\
                                                                          'Contract ID': 'count'})
healthdfsupplier = healthdfsupplier.reset_index().rename(columns = {'Contract ID': 'No. of Contracts'})

top10healthsupplier = healthdfsupplier.nlargest(10, columns = 'Applicable Value')

#Groupby and sort for the top ten divisions:
healthdfdiv = healthdf.groupby(['Agency Name', 'Division']).agg({'Applicable Value': 'sum',\
                                                                 'Contract ID': 'count'})
healthdfdiv = healthdfdiv.reset_index().rename(columns = {'Contract ID': 'No. of Contracts'})

top10healthdiv = healthdfdiv.nlargest(10, columns = 'Applicable Value')

#Groupby and sort for the top supplier states:
healthdfstate = healthdf.groupby(['Agency Name', 'Supplier State']).agg({'Applicable Value': 'sum',\
                                                                         'Contract ID': 'count'})
healthdfstate = healthdfstate.reset_index().rename(columns = {'Contract ID': 'No. of Contracts'})

tophealthstate = healthdfstate.nlargest(10, columns = 'Applicable Value')

#Convert start date to datetime format
healthdf['ANAO_Applicable Publish Date'] = healthdf['Applicable Publish Date'].apply(lambda x: \
                                           datetime.strptime(x, '%d/%m/%Y'))

#Groupby Applicable Publish data, sum applicable value and get count of contracts
healthdftimeseries = healthdf.groupby(['ANAO_Applicable Publish Date']).agg({'Applicable Value': 'sum',\
                                      'Contract ID': 'count'}).reset_index()

#Uncomment to process matplotlib in console
#%matplotlib inline

plt.rc('font', size=12)
fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False, figsize=(20, 12))

#Specify values for lines in the timeseries
ax[0].plot(healthdftimeseries['ANAO_Applicable Publish Date'], healthdftimeseries['Applicable Value'],\
        color = 'mediumblue', label = 'Total Contract Value')
ax[1].plot(healthdftimeseries['ANAO_Applicable Publish Date'], healthdftimeseries['Contract ID'],\
        color = 'cornflowerblue', label = 'No. of Contracts')

#Set the x-axis to do major ticks on the months and label them like 'Jul-19'
for i in [0,1]:
    ax[i].xaxis.set_major_locator(MonthLocator())
    ax[i].xaxis.set_major_formatter(DateFormatter('%b-%y'))

#Set the y-axis to have currency format like '$000,000,000'
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax[0].yaxis.set_major_formatter(tick)

#Set reference line for COVID-19 Government Pandemic Response date:
for i in [0,1]:
    ax[i].axvline(x = datetime(2020,3,15), color = 'red', label = 'Start of Commonwealth response to' + \
                  ' COVID-19')
    
#Time series formatting
ax[0].set_xlabel('Publish Date')
ax[0].set_ylabel('Aggregate Contract Value ($)')
ax[0].set_title('2019-20 FY Contract Value Committed')
ax[0].grid(True)
ax[0].legend(loc = 'upper left')

ax[1].set_xlabel('Publish Date')
ax[1].set_ylabel('No. of Contracts')
ax[1].set_title('2019-20 FY No. of Contracts Committed')
ax[1].grid(True)
ax[1].legend(loc = 'upper left')

fig

#%%
#Export files
os.chdir(askdirectory(title = 'Selecting output directory'))

with pd.ExcelWriter('2019-20 SADA Insights - Department of Health.xlsx') as writer:
    top10limitedtenderamt.to_excel(writer, sheet_name = 'Top 10 Agencies', index = False)
    top10healthdesc.to_excel(writer, sheet_name = 'Top 10 Procurement Desc', index = False)
    top10healthcat.to_excel(writer, sheet_name = 'Top 10 Procurement Cat', index = False)
    top10healthsupplier.to_excel(writer, sheet_name = 'Top 10 Suppliers', index = False)
    top10healthdiv.to_excel(writer, sheet_name = 'Top 10 Divisions', index = False)
    tophealthstate.to_excel(writer, sheet_name = 'Top 10 Supplier States', index = False)
    