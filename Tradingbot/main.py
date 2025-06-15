import os
import requests
import pandas as pd
import json
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go


instrument = 'AUD_USD'
url = 'https://api-fxpractice.oanda.com/v3/instruments/{}/candles'.format(instrument)


headers = {
    'Content-Type' : 'application/json',
    'Authorization' : 'Bearer {}'.format(os.environ['token']),
    'AcceptDatetimeFormat' : 'RFC3339'
}

params = {
    'granularity' : 'D',
    'from' : (datetime.now(timezone.utc)-timedelta(days=80)).astimezone().isoformat(),
    'to' : datetime.now(timezone.utc).astimezone().isoformat()
}

response = requests.get(url, headers = headers, params=params)
#print(response.status_code)
jsonresult = json.loads(response.text)

output = pd.DataFrame()
for idx, daily in enumerate(jsonresult['candles']):
    to_add = {'date' : daily['time'],
              'open' : daily['mid']['o'],
              'high' : daily['mid']['h'],
              'low'  : daily['mid']['l'],
              'close': daily['mid']['c'],
              'volume' : daily['volume'],
              'complete' : daily['complete']}
    
    output = pd.concat([output, pd.DataFrame(to_add, index=[idx])])

#Format
for col in ['open', 'high', 'low', 'close']:
    output[col] = output[col].astype(float)

output['date'] = pd.to_datetime(output['date'])

#Visual
fig = go.Figure(data=[go.Candlestick(
    x=output['date'],
    open=output['open'],
    high=output['high'],
    low=output['low'],
    close=output['close'],
    increasing_line_color='green',
    decreasing_line_color='red'
)])

# Update layout
fig.update_layout(
    title='AUD/USD 1 day Candlestick Chart (last 60 days)',
    xaxis_title='Time',
    yaxis_title='Exchange Rate',
    xaxis_rangeslider_visible=False,
    template='plotly_dark'
)

# Show chart
fig.show(renderer = 'browser')

#Function (to be moved)
def isbetween(value, boundary1, boundary2) -> bool:
    return ((value < boundary2) & (value > boundary1)) | ((value < boundary1) & (value > boundary2))

def swinglow(value, adjvalue1, adjvalue2) -> bool:
    return ((value < adjvalue2) & (value < adjvalue1))

def swinghigh(value, adjvalue1, adjvalue2) -> bool:
    return ((value > adjvalue2) & (value > adjvalue1))

#Algorithm
n = output.shape[0]
dates = output['date']
lows = output['low']
highs = output['high']
open = output['open']
close = output['close']

#Simple search
search_idx = n-2
for idx in reversed(range(2, search_idx)):
    #Identify swing low or swing high
    if swinglow(lows[idx-1], lows[idx-2], lows[idx]):
        if isbetween(lows[idx-1], open[search_idx], close[search_idx]) & (close[search_idx] > lows[search_idx-1]):
            print(f"Condition found on {dates[idx-1].strftime('%d-%m-%Y')}: liquidity point {lows[idx-1]}")
            break
    elif swinghigh(highs[idx-1], highs[idx-2], highs[idx]):
        if isbetween(highs[idx-1], open[search_idx], close[search_idx]) & (close[search_idx] < highs[search_idx-1]):
            print(f"Condition found on {dates[idx-1].strftime('%d-%m-%Y')}: liquidity point {lows[idx-1]}")
            break


#TODO
#Full search
for idx in reversed(range(2, n-2)):
    #Identify swing low or swing high
    if swinglow(lows[idx-1], lows[idx-2], lows[idx]) | swinghigh(lows[idx-1], lows[idx-2], lows[idx]):
        for search_idx in reversed(range(idx-3)):
            print(search_idx)
            #Search for cut through low & closed inside previous candle
            if isbetween(lows[idx+1], open[search_idx], close[search_idx]) & (close[search_idx] > lows[search_idx-1]):
                print(f"Condition found on {dates[idx-1].strftime('%d-%m-%Y')}: liquidity point {lows[idx+1]}")
                break


def search_condition(highs, lows, open, close, full_search:bool):
    n = highs.shape[0]
    for idx in range(n-2):
        #Identify swing low or swing high
        if ((lows[idx+1] < lows[idx+2]) & (lows[idx+1] < lows[idx])):
            for search_idx in range(idx+3, n):
                #Search for cut through low & closed inside previous candle
                if isbetween(lows[idx+1], open[search_idx], close[search_idx]) & (close[search_idx] > lows[search_idx-1]):
                    print(f"Condition found: liquidity {lows[idx+1]} \nsearched {close[search_idx]}")
                    #return lows[idx+1]
