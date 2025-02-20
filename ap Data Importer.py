import MetaTrader5 as mt5
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
login =
password =
server =
mt5.initialize()
mt5.login(login, password, server) # start program and login # login and intialize
# -----------------------------------------------------------------------------------------------
# Variables
ticker = 'USDMXN'
time1 = datetime.datetime(2024, 1, 1)
time2 = datetime.datetime(2024, 9, 1)
hours_in_trading_day = 24

# -----------------------------------------------------------------------------------------------
# Derivatives
td = hours_in_trading_day * 60 * 60
nd = (time2.timestamp() - time1.timestamp())/td
print(f'You are analyzing {ticker} for {nd} number of days.')
print(f'Your start data is {time1}.')
print(f'Your end date is {time2}.')
print()

utime1 = int(time1.timestamp())
utime2 = int(time2.timestamp())

# Data frame for ticker
tick_data = pd.DataFrame(mt5.copy_ticks_range(ticker, time1, time2, mt5.COPY_TICKS_ALL))
print('Tick data import was successful')
print(tick_data)
td_bid_ask = pd.DataFrame({'time': tick_data['time'], 'bid': tick_data['bid'], 'ask': tick_data['ask']})
processed_td_bid_ask = td_bid_ask.groupby('time').mean().reset_index()
td = processed_td_bid_ask.set_index('time').reindex(range(utime1, utime2+1)).interpolate().reset_index()
td.fillna(0, inplace=True)
print('Tick data was successfully processed')
print()


# Remove weekends
print(f'There are {len(td)} seconds to iterate over in this sample')
print('Removing Weekends...')

def is_between_friday_to_sunday(timestamp):
    # Convert timestamp to datetime object in UTC
    dt = datetime.datetime.utcfromtimestamp(timestamp)

    # Get the day of the week (0 = Monday, 6 = Sunday)
    day_of_week = dt.weekday()

    # Check if it's after 21:00 UTC on Friday and before 21:00 UTC on Sunday
    if (day_of_week == 4 and dt.hour >= 21) or (day_of_week == 6 and dt.hour < 21):
        return True
    # Check if it's Saturday (day_of_week == 5)
    elif day_of_week == 5:
        return True
    else:
        return False

ave_bid =  sum(td['bid']) / len(td['bid'])
ave_ask = sum(td['ask']) / len(td['ask'])
ave = (ave_bid + ave_ask)/2
for index, row in tqdm(td.iterrows()):
    if is_between_friday_to_sunday(row['time']) == True:
        # Set bid and ask columns to 0
        td.at[index, 'bid'] = 0
        td.at[index, 'ask'] = 0
print('Weekends successfully removed')
print()

# Creating ap column
td['average price'] = (td['bid'] + td['ask'])/2
ap = td
print('ap successfully created')
print()

# Saving data as .csv
directory = 'ap Data'
time1_str = time1.strftime('%Y-%m-%d %H-%M-%S')
time2_str = time2.strftime('%Y-%m-%d %H-%M-%S')

file_path = os.path.join(directory, f'ap({ticker}, {time1_str}, {time2_str})_data.csv')
ap.to_csv(file_path, index=False)

print(f'ap Data for {ticker} successfully saved for {time1} to {time2}')
