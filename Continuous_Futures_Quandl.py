# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:48:11 2017

@author: Colton Smith
"""

import pandas as pd
import numpy as np
import quandl
import matplotlib.pyplot as mp
import math as math
from matplotlib.backends.backend_pdf import PdfPages

def open_interest_perp(df_dict, perp_len):    
    start = np.datetime64('2009-12-01')
    end = np.datetime64('2017-12-01')
    date_list = np.arange(start,end)
    con_dates = pd.DataFrame(index = date_list)
    con_dates['con_num'] = 0

    for i in range(0, len(df_dict)-1,1):
        print(i)
        cur_con = df_dict[contracts[i]]
        next_con = df_dict[contracts[i+1]]
        
        sub = next_con.merge(cur_con,how='inner',left_index=True,right_index=True)
        sub = sub[sub.index.values > start]
        sub = sub[sub['Open Interest_x'] > sub['Open Interest_y']]
        start = sub.index.values[0]
        con_dates['con_num'] = np.where(con_dates.index.values >= start, con_dates['con_num'] + 1, con_dates['con_num'])    
        get_start = sub[sub['Open Interest_x'] == max(sub['Open Interest_x'])]
        start = get_start.index.values[0]
        
        sub = sub[['Price_x','Open Interest_x','Price_y','Open Interest_y']]
        sub[['PPrice']] = sub[['Price_x']]
        for j in range(0,perp_len):
            tot_int = sub['Open Interest_x'][j] + sub['Open Interest_y'][j]
            sub.PPrice[j] = (sub.Price_x[j]*sub['Open Interest_x'][j] + sub.Price_y[j]*sub['Open Interest_y'][j])/tot_int
            next_con.Price = np.where(next_con.index.values == sub.index.values[j], sub.PPrice[j], next_con.Price)
            df_dict[contracts[i+1]] = next_con

    con_dates_ind = con_dates.drop_duplicates()
    
    for i in range(0, len(df_dict)-1,1):
        cur_con = df_dict[contracts[i]]
        remove = con_dates_ind.index.values[con_dates_ind.con_num == (i+1)]
        df_dict[contracts[i]] = cur_con[cur_con.index.values < remove]
     
    return df_dict

################################################## Main ###################################################
    
### Initialize Variables ###
CME_months = ['H','K','N','U','Z']
CME_years = ['2010','2011','2012','2013','2014','2015','2016','2017']
ticker = 'W'
quandl_api = "" ### Insert your Quandl API key
df_dict = {}
contracts = []

### Get contract data from Quandl ###
for i in range(0,len(CME_months)*len(CME_years)):
    month = CME_months[i % len(CME_months)]
    year = CME_years[math.floor(i/len(CME_months))]
    print('Downloading: ' + month + ' ' + year)
    contract = month + year
    contracts.append(contract)
    current_df = quandl.get('CME/' + ticker + contract, authtoken = quandl_api)
    current_df = current_df[[current_df.columns[5],current_df.columns[6],current_df.columns[7]]]
    current_df.columns = ['Price','Volume','Open Interest']
    df_dict[contract]= current_df

### Use open interest instead of expiration for rolling over? ###
### Perpetual, how many days? ###
# df_dict = open_interest_perp(df_dict,5) ### 5-day Perpetual
# df_dict = open_interest_perp(df_dict,0) ### Open Interest forward/backward 
#################################################################
    
### Fencepost ###
start = np.datetime64('2009-12-01')
cont = df_dict[contracts[0]]
cont = cont[cont.index.values >= start]
start = cont.index.values[len(cont)-1]
switch_post = pd.DataFrame(columns = ('Price','Volume','Open Interest'))
switch_pre = pd.DataFrame(columns = ('Price','Volume','Open Interest'))
switch_pre = switch_pre.append(cont.iloc[len(cont)-1])

for i in range(1,len(CME_months)*len(CME_years)):
    cur = df_dict[contracts[i]]
    cur = cur[cur.index.values > start]
    switch_post = switch_post.append(cur.iloc[0])
    start = cur.index.values[len(cur)-1]
    switch_pre = switch_pre.append(cur.iloc[len(cur)-1])
    cont = cont.append(cur)
    
switch_pre = switch_pre.drop(switch_pre.index[len(switch_pre)-1])
switch_post['change'] = switch_post['Price'].values - switch_pre['Price'].values
switch_post = pd.DataFrame(switch_post['change'])

main = cont.merge(switch_post,how='left',left_index=True,right_index=True)
main = main.fillna(0)
main['forward'] = main.change.cumsum()
main['backward'] = main.loc[::-1, 'change'].cumsum()[::-1]
main.backward = main.backward.shift(-1)
main = main.fillna(0)

main['Forward Adjusted'] = main.Price - main.forward
main['Backward Adjusted'] = main.Price + main.backward
main['diff'] = main['Forward Adjusted'] - main['Backward Adjusted'] 

### Price Series ###
main = main.rename(columns={'Price': 'Unadjusted'})

### Set up for backward/forward adjustments ###
output = PdfPages('Current.pdf')

mp.plot(main['Unadjusted'], linewidth = 0.75)
mp.plot(main['Forward Adjusted'], linewidth = 0.75)
mp.plot(main['Backward Adjusted'], linewidth = 0.75)
mp.legend(loc='upper right')
mp.ylabel('Price')
mp.title('Wheat: Expiration Rollover')
mp.grid(True, linewidth = 0.25)

output.savefig()
output.close()
mp.close()
