import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import time
import matplotlib.pyplot as plt

api_key = '' # AlphaVantage API key
ETFs = np.array(['XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLK','XLU']) # all ETFs used

ts = TimeSeries(key = api_key, output_format = 'pandas', indexing_type='date')
    
price_data = pd.DataFrame()
# Loop through requested securities
print('Querying Securities, Estimated time: ' + str(round(len(ETFs)/5)) + ' minutes')
for x in range(len(ETFs)):
    print(str(ETFs[x]))

    if (x + 1) % 5 == 0:
        time.sleep(60) # wait 1 minutes after every 5 API calls

    data, meta_data = ts.get_weekly_adjusted(symbol=str(ETFs[x])) 
    data = data['5. adjusted close'].iloc[::-1]             
    data = pd.DataFrame(data).rename(index=str, columns={'5. adjusted close' : str(ETFs[x])})
    price_data = pd.concat([price_data,data], axis=1, sort=False)
    
price_data = price_data.iloc[:300].iloc[::-1]

plt.plot(price_data)
plt.xticks([])

### Calculate Returns ###
returns_data = price_data.copy()
returns_data = returns_data.apply(func = lambda x: x.shift(-1)/x - 1, axis = 0)

### Calculate Momentum Signal ###
momentum_signal = price_data.copy()
momentum_signal = momentum_signal.apply(func = lambda x: x.shift(1)/x.shift(7) - 1, axis = 0)
momentum_signal = momentum_signal.rank(axis = 1)

for col in momentum_signal.columns:
    momentum_signal[col] = np.where(momentum_signal[col] >= 8, 1, np.where(momentum_signal[col] <= 2, -1, 0))

returns_signals = np.multiply(returns_data, momentum_signal)
portf_returns = pd.DataFrame(index = momentum_signal.index, columns = ['ls'])
portf_returns = returns_signals.sum(axis = 1) / 4
portf_cum_returns = np.exp(np.log1p(portf_returns).cumsum())

plt.plot(portf_cum_returns)
plt.xticks([])

