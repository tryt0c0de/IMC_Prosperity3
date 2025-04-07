import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.read_csv('../Data/trade_history.csv')

df_kelp = df[df['symbol'] == 'KELP']

tot_quant_kelp = {time: df_kelp[df_kelp['timestamp']==time]['quantity'].sum() for time in df_kelp['timestamp'].unique()}

df_kelp['tot_quant'] = [tot_quant_kelp[time] for time in df_kelp['timestamp']]
df_kelp['w_price'] = df_kelp['price'] * df_kelp['quantity'] / df_kelp['tot_quant']

data_kelp = df_kelp.groupby('timestamp')[['w_price']].agg('sum')



span = 20
u_bound = 1/1000

data_kelp[f'EWM_{span}'] = data_kelp['w_price'].ewm(span=span, adjust=False).mean()
data_kelp[f'spread_EWM_{span}'] = (data_kelp['w_price'] - data_kelp[f'EWM_{span}']) / data_kelp[f'EWM_{span}']
data_kelp[f'move_price_{span}'] = (data_kelp['w_price'] - data_kelp['w_price'].shift(1)).fillna(0)

sell = data_kelp[f'spread_EWM_{span}'] > u_bound
buy = data_kelp[f'spread_EWM_{span}'] < -u_bound

hold_quant = 0
pnl = 0

for i, row in data_kelp.iterrows():
    pnl += row[f'move_price_{span}'] * hold_quant
    if sell[i] or buy[i]:
        hold_quant -= row[f'spread_EWM_{span}']



'''plt.plot(data_kelp['w_price'], label='price')
plt.plot(data_kelp[f'EWM_{span}'], label = 'ewm')

sell = data_kelp[f'spread_EWM_{span}'] > 1/1000
buy = data_kelp[f'spread_EWM_{span}'] < -1/1000

i_sell = data_kelp[sell]
i_buy = data_kelp[buy]

plt.scatter(i_sell.index, i_sell['w_price'], color='red', s=20000*i_sell[f'spread_EWM_{span}'])
plt.scatter(i_buy.index, i_buy['w_price'], color='green', s=-20000*i_buy[f'spread_EWM_{span}'])
'''

plt.plot(data_kelp[f'spread_EWM_{span}'])

plt.legend()
plt.show()

