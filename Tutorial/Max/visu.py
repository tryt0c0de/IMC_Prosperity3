import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.read_csv('../Data/trade_history.csv')

df_kelp = df[df['symbol'] == 'KELP']
#df_kelp = df_kelp[df_kelp['timestamp'] < 30000]
df_resin = df[df['symbol'] != 'KELP']

tot_quant_kelp = {time: df_kelp[df_kelp['timestamp']==time]['quantity'].sum() for time in df_kelp['timestamp'].unique()}
tot_quant_resin = {time: df_resin[df_resin['timestamp']==time]['quantity'].sum() for time in df_resin['timestamp'].unique()}

df_kelp['tot_quant'] = [tot_quant_kelp[time] for time in df_kelp['timestamp']]
df_kelp['w_price'] = df_kelp['price'] * df_kelp['quantity'] / df_kelp['tot_quant']

df_resin['tot_quant'] = [tot_quant_resin[time] for time in df_resin['timestamp']]
df_resin['w_price'] = df_resin['price'] * df_resin['quantity'] / df_resin['tot_quant']


data_kelp = df_kelp.groupby('timestamp')[['w_price']].agg('sum')
data_resin = df_resin.groupby('timestamp')['w_price'].agg('sum')








range_span = range(21,22)

for span in range_span:
    data_kelp[f'EWM_{span}'] = data_kelp['w_price'].ewm(span=span, adjust=False).mean()
    data_kelp[f'spread_EWM_{span}'] = (data_kelp['w_price'] - data_kelp[f'EWM_{span}']) / data_kelp[f'EWM_{span}']
    data_kelp[f'move_price_{span}'] = (data_kelp['w_price'] - data_kelp['w_price'].shift(1)).fillna(0)

def get_pnl(u_bound, span):

    sell = data_kelp[f'spread_EWM_{span}'] > u_bound
    buy = data_kelp[f'spread_EWM_{span}'] < -u_bound

    hold_quant = 0
    pnl = 0

    for i, row in data_kelp.iterrows():
        pnl += row[f'move_price_{span}'] * hold_quant
        if sell[i] or buy[i]:
            hold_quant -= row[f'spread_EWM_{span}']
    return pnl

max_pnl = 0
best_u = 0
best_span = 0

for span in tqdm(range_span):
    for u_bound in np.linspace(1/10000,1/10000, 1):
        pnl = get_pnl(u_bound, span)
        if pnl > max_pnl:
            max_pnl = pnl
            best_u = u_bound
            best_span = span

print(max_pnl)
print(best_u)
print(best_span)

plt.plot(data_kelp['w_price'], label='price')
plt.plot(data_kelp['EWM_21'], label = 'ewm')

sell = data_kelp[f'spread_EWM_{21}'] > 1/1000
buy = data_kelp[f'spread_EWM_{21}'] < -1/1000

i_sell = data_kelp[sell]
i_buy = data_kelp[buy]

plt.scatter(i_sell.index, i_sell['w_price'], color='red', s=20000*i_sell['spread_EWM_21'])
plt.scatter(i_buy.index, i_buy['w_price'], color='green', s=-20000*i_buy['spread_EWM_21'])


plt.legend()
plt.show()

