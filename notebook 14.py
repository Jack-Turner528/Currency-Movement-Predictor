import numpy as np
from datetime import datetime
import os
import pandas as pd
from scipy.stats import linregress
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
# -----------------------------------------------------------------------------------------------
def importap(ticker,time1,time2,hours_in_trading_day,atime1,atime2):
    def cdf(df, x, y):
        return df[(df['time'] >= x) & (df['time'] <= y)]
    directory = 'New ap Data'
    time1_str = time1.strftime('%Y-%m-%d %H-%M-%S')
    time2_str = time2.strftime('%Y-%m-%d %H-%M-%S')
    ap_file_path = os.path.join(directory, f'ap({ticker}, {time1_str}, {time2_str})_data.csv')
    if os.path.exists(ap_file_path):
        print('ap file can be accessed')
    else:
        print("ap file can't be accessed")
    ap = pd.read_csv(ap_file_path)
    ap.index = range(1, len(ap) + 1)  # Importing ap
    print(f'ap was imported for ({ticker}, {time1_str}, {time2_str})')
    uatime1 = int(atime1.timestamp())
    uatime2 = int(atime2.timestamp())
    aP = cdf(ap, uatime1, uatime2)
    td = hours_in_trading_day * 60 * 60
    nd = (atime2.timestamp() - atime1.timestamp()) / td
    ap_ave = pd.DataFrame({'time': ap['time'], 'average price': ap['average price']})
    aP_ave = pd.DataFrame({'time': aP['time'], 'average price': aP['average price']})
    print(f'ap was cut to be from {atime1} to {atime2}. There are {nd} days in this sample.')
    print()
    return ap, aP, td, ap_ave, aP_ave
def CA(v, N):
    v = (
        v.groupby(data.index // N)  # Group indices into chunks of N
        .mean()  # Calculate the mean for each chunk
        .to_frame()  # Convert to DataFrame and rename the column
    )
    return v
def average_previous_tf_values(v, tf):
    # Calculate the indices to average over
    indices = [(i + 1) * (tf - 1) for i in range(len(v) // (tf - 1))]
    # Create a list to store the averaged values
    averaged_values = [v.iloc[max(0, idx - tf + 1): idx + 1].mean() for idx in indices]
    # Create a new DataFrame with the averaged values
    new_df = pd.DataFrame(averaged_values, columns=['Averaged Values'])
    return new_df
def EVAI(v, tf):
    # Calculate the indices to extract
    indices = [(i + 1) * (tf - 1) for i in range(len(v) // (tf - 1))]
    # Create a new DataFrame with the extracted values
    new_df = pd.DataFrame(v.iloc[indices].values, columns=['Extracted Values'])
    return new_df
def pddx(v, N):
    v = v.iloc[:, 0]
    # Initialize the output vector u as a float series
    u = pd.Series([0.0] * len(v), index=v.index)

    # Iterate over each index in the vector
    for i in range(1, len(v)):
        # Only calculate if there are at least N values
        if i >= N - 1:
            # Get the last N values
            v_subset = v[i - N + 1:i + 1]

            # Perform linear regression on the subset
            x = np.arange(len(v_subset))
            slope, _, _, _, _ = linregress(x, v_subset)
            u[i] = slope
        else:
            u[i] = 0.0  # Set to 0 if less than N values are available

    return u
def pp(v, buffer):
    # ------------------------------------------------------------------------
    # Variable intialize
    a = 0  # buffer counter
    c = 0  # hold counter
    b = 0 # buy counter
    Ppt = 0  # Total profit
    k = 0
    buy_vector = []
    sell_vector = []
    Pp_vector = []
    #------------------------------------------------------------------------
    for index, row in v.iterrows():
        z = row['signal']  # Signal (buy/sell)
        x = row['bid']  # Bid price (buy (reversed because you start owning the asset))
        y = row['ask']  # Ask price (sell)
        if a != 0:
            a += -1
        if c == 0 and z < 0 and a == 0:
            b = x
            c = 1
            a = buffer
            buy_vector.append(index)
        elif c == 1 and z >= 0 and a == 0:
            Pp = (b/y) - 1
            Ppt += Pp
            Pp_vector.append(Ppt)
            a = buffer
            c = 0
            b = 0
            sell_vector.append(index)
            k += 1
            continue
        if c == 0:
            Pp_vector.append(Ppt)
        if c == 1:
            Pp_vector.append(Ppt + (b/y) - 1)

    if c == 1:
        y = v['ask'].iloc[-1]
        Pp = (b / y) - 1
        Ppt += Pp
        print(f'last forced buy = {Pp}')
    return Ppt, buy_vector, sell_vector, k, Pp_vector
def pp1(v, buffer):
    # ------------------------------------------------------------------------
    # Variable intialize
    a = 0  # buffer counter
    c = 0  # hold counter
    b = 0  # buy counter
    Ppt = 1  # Total profit
    k = 0
    A = 1
    buy_vector = []
    sell_vector = []
    Pp_vector = []
    # ------------------------------------------------------------------------
    for index, row in v.iterrows():
        z = row['signal']  # Signal (buy/sell)
        x = row['bid']  # Bid price (buy (reversed because you start owning the asset))
        y = row['ask']  # Ask price (sell)
        if a != 0:
            a += -1
        if c == 0 and z < 0 and a == 0:
            b = x
            c = 1
            a = buffer
            buy_vector.append(index)
        elif c == 1 and z >= 0 and a == 0:
            A = A * (b / y)
            a = buffer
            c = 0
            b = 0
            sell_vector.append(index)
            k += 1
        Pp_vector.append(A - 1)
    if c == 1:
        y = v['ask'].iloc[-1]
        A * (b / y)
        print(f'last forced buy = {b/y} %')
    return A - 1, buy_vector, sell_vector, k, Pp_vector
def pp2(v, buffer):
    # ------------------------------------------------------------------------
    # Variable intialize
    a = 0  # buffer counter
    c = 0  # hold counter
    b = 0  # buy counter
    Ppt = 0  # Total profit
    k = 0
    buy_vector = []
    sell_vector = []
    Pp_vector = []
    # ------------------------------------------------------------------------
    for index, row in v.iterrows():
        z = row['signal1']  # Signal (buy/sell)
        z1 = row['signal2']
        x = row['bid']  # Bid price (buy (reversed because you start owning the asset))
        y = row['ask']  # Ask price (sell)
        if a != 0:
            a += -1
        if c == 0 and z < 0 and z1 < 0 and a == 0:
            b = x
            c = 1
            a = buffer
            buy_vector.append(index)
        elif c == 1 and z >= 0 and a == 0:
            Pp = (b / y) - 1
            Ppt += Pp
            a = buffer
            c = 0
            b = 0
            sell_vector.append(index)
            k += 1
        Pp_vector.append(Ppt)
    if c == 1:
        y = v['ask'].iloc[-1]
        Pp = (b / y) - 1
        Ppt += Pp
        print(f'last forced buy = {Pp}')
    return Ppt, buy_vector, sell_vector, k, Pp_vector
def pp3(v, buffer):
    # ------------------------------------------------------------------------
    # Variable intialize
    A = 1
    a1 = buffer  # buffer counter
    a2 = 0 #buffer
    c = 0  # hold counter
    b = 0 # buy counter
    Ppt = 0  # Total profit
    k = 0
    buy_vector = []
    sell_vector = []
    Pp_vector = []
    #------------------------------------------------------------------------
    for index, row in v.iterrows():
        z = row['signal']  # Signal (buy/sell)
        x = row['bid']  # Bid price
        y = row['ask']  # Ask price
        if c == 0:  # Only check for buys when not holding an asset
            if z < 0 and a1 != 0:
                a1 -= 1  # Decrement buffer if signal is negative
            elif z >= 0:
                a1 = buffer  # Reset buffer if signal is not negative

            if a1 == 0 and z < 0:  # Buy condition met
                b = x
                c = 1  # Set holding position
                a1 = buffer  # Reset buffer after buy
                print(f'buy at {b} at {index}, A = {A}')
                buy_vector.append(index)
        if c == 1:
            if z >= 0 and a2 != 0:
                a2 -= 1  # Decrement buffer if signal is negative
            elif z < 0:
                a2 = 0 # buffer  # Reset buffer if signal is not negative

            if z >= 0 and a2 == 0:  # Sell condition
                Pp = (b / y)
                A1 = A
                A = A * Pp
                Ppt += Pp - 1
                c = 0  # Reset holding position
                b = 0
                print(f'sell at {y} at {index}, A = {A}')
                print(f'P = {A - A1}')
                A1 = 0
                print()
                sell_vector.append(index)
                a2 = 0 #buffer
                k += 1
        if y == 0 or b == 0:
            Pp_vector.append(A - 1)
        else:
            Pp_vector.append(b/y * A - 1)

    if c == 1:
        y = v['ask'].iloc[-1]
        Pp = (b / y)
        A = A * Pp
        print(f'sell at {y}, A = {A}')
    print(f'A = {A}')
    print(f'A - 1 = {A - 1}')
    return A - 1, buy_vector, sell_vector, k, Pp_vector
def SMA(df, column_name, N):
    """
    Applies a simple moving average (SMA) of size N to a specified column in the DataFrame,
    and returns a DataFrame with the SMA values while keeping the same size.

    The first N-1 values are set to 0.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to apply the SMA on.
    N (int): The window size for the moving average.

    Returns:
    pd.DataFrame: The modified DataFrame with SMA values and the first N-1 values set to 0.
    """
    # Copy the input DataFrame to avoid modifying the original
    ndf = df.copy()

    # Calculate the rolling mean
    ndf[column_name] = ndf[column_name].rolling(window=N).mean()

    # Fill the first N-1 entries with 0
    ndf[column_name] = ndf[column_name].fillna(0)

    return ndf

def plot_values_at_indices(df, column_name, indices, color, m, N, a):
    """
    Plots the values in the specified DataFrame column at given indices as points.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to plot.
    indices (list): A list of indices where values should be plotted as points.
    color (str): Color for the points in the scatter plot.

    Returns:
    None: Displays a scatter plot of the specified points.
    """
    # Calculate the actual indices based on m and N
    calculated_indices = [(value * m) - (N * m) for value in indices]

    # Check if calculated indices should be treated as positional or label-based indices
    try:
        # Use .loc if calculated_indices are labels
        values = df.loc[calculated_indices, column_name]
    except KeyError:
        # Use .iloc if calculated_indices are positions
        values = df.iloc[calculated_indices][column_name]

    # Plotting the points
    a.scatter(calculated_indices, values, color=color, label='Points at Indices', zorder=10)

    # Optional plot formatting
    '''
    a.set_xlabel('Index')
    a.set_ylabel(column_name)
    a.set_title(f'Values in {column_name} at Specified Indices')
    a.legend()
    '''
def create_boolean_vector(u):
    """
    Creates a new DataFrame vector v, where v[i] is True if u[i] < 0, and False otherwise.

    Parameters:
    u (pd.Series or pd.DataFrame): A single-column DataFrame or Series vector.

    Returns:
    pd.Series: A vector of the same size as u, containing boolean values.
    """
    if isinstance(u, pd.DataFrame):
        # Ensure it's a single-column DataFrame
        if u.shape[1] != 1:
            raise ValueError("Input DataFrame must have only one column.")
        u = u.iloc[:, 0]  # Extract the single column as a Series

    # Compute the boolean vector
    v = u < 0
    return v
def plot_colored_graph(df):
    """
    Plots the first column of a DataFrame as a line plot, coloring the line green
    when the second column is True, and red when it is False.

    Parameters:
    df (pd.DataFrame): A DataFrame with two columns. The first column contains numbers,
                       and the second contains boolean values (True/False).
    """
    # Extract the columns
    y = df.iloc[:, 0]
    color_condition = df.iloc[:, 1]

    # Set up the plot
    plt.figure(figsize=(10, 5))

    # Loop through segments and plot with respective colors
    start_idx = 0
    for idx in range(1, len(df)):
        # When the condition changes or at the end, plot the segment
        if color_condition[idx] != color_condition[start_idx] or idx == len(df) - 1:
            # Determine the color based on the condition
            color = 'green' if color_condition[start_idx] else 'red'
            # Extend to the last index if we're at the end
            end_idx = idx if idx == len(df) - 1 else idx - 1
            # Plot the segment
            plt.plot(range(start_idx, end_idx + 1), y[start_idx:end_idx + 1], color=color)
            # Update the start index
            start_idx = idx

    plt.title("Colored Line Graph")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
# -----------------------------------------------------------------------------------------------
# stuff to import
ticker = 'USDMXN'
time1 = datetime(2024, 7, 1)
time2 = datetime(2024, 11, 1)
atime1 = datetime(2024, 7, 1)
atime2 = datetime(2024, 11, 1)
F = 8/100000
N1 = 360
N2 = 360
N3 = 1
N4 = 2
R = 0*10**-6
thrs = 360 # 360
tf = 60
plot_SMA_and_dx = True
plot_real_vs_SMA = True
plot_profit = False
graph_real_color = True
graph_SMA_color = True
dx_vs_dx2 = False
# -----------------------------------------------------------------------------------------------
# Getting aP
ap, aP, td, ap_ave, aP_ave = importap(ticker, time1, time2, 24, atime1, atime2)

# Getting bid, ask, and aP0
aP0 = aP[aP['average price'] != 0]
ndata = aP0['average price'].reset_index(drop=True)
bid = EVAI(aP0['bid'], tf)
ask = EVAI(aP0['ask'], tf)
data = EVAI(aP0['average price'], tf)
print('Data sliced successfully')

# Applying SMA to data
sdata = SMA(data, 'Extracted Values', N1)
print('SMA done successfully')

if plot_real_vs_SMA:
    plt.plot(sdata.iloc[N1:], zorder = 9)
    plt.plot(bid)
    plt.title('Real Data vs SMA')
    plt.show()

# Calculating the pseduo-derivative
pd_sdata_dx = pddx(sdata, N2)
pd_sdata_dx = pd.DataFrame(pd_sdata_dx)
pd_sdata_dx = pd_sdata_dx[0]/sdata['Extracted Values']
pd_sdata_dx.fillna(0, inplace=True)
pd_sdata_dx = pd.DataFrame(pd_sdata_dx)
print(pd_sdata_dx)
pd_sdata_dx = pd_sdata_dx.applymap(lambda x: 0 if 0 > x > R else x)
print('Slope calculated successfully')
print()

# Calculating the 2pseduo-derivative
#p2d_sdata_d2x = pddx(pd_sdata_dx, N2)
#p2d_sdata_d2x = pd.DataFrame(p2d_sdata_d2x)
#p2d_sdata_d2x = p2d_sdata_d2x * 10**3.5
#print('Slope calculated successfully')
#print()

# Combining Dataframes
combined_df = pd.concat([bid, ask, pd_sdata_dx], axis=1)
combined_df.columns = ['bid', 'ask', 'signal']
Ppt, buy_vector, sell_vector, k, Pp_vector = pp3(combined_df, thrs)
print(f'The Profit would be {Ppt - k * F}')
# -----------------------------------------------------------------------------------------------
if graph_real_color:
    Npd_sdata_dx = pd_sdata_dx.iloc[N1 + N2:]
    B = create_boolean_vector(Npd_sdata_dx)
    Nsdata = sdata.iloc[N1 + N2:]
    data = pd.concat([bid.iloc[N1 + N2:], B], axis=1)
    data = data.reset_index(drop=True)
    plot_colored_graph(data)
if graph_SMA_color:
    Npd_sdata_dx = pd_sdata_dx.iloc[N1 + N2:]
    B = create_boolean_vector(Npd_sdata_dx)
    Nsdata = sdata.iloc[N1 + N2:]
    data = pd.concat([Nsdata, B], axis=1)
    data = data.reset_index(drop = True)
    plot_colored_graph(data)
if plot_profit:
    fig2, ax1 = plt.subplots()
    Pp_vector = pd.DataFrame(Pp_vector)
    ax1.plot(Pp_vector.iloc[N3:])
    # plot_values_at_indices(Pp_vector, 0, sell_vector, 'r', 1, 0, ax1)
    # plot_values_at_indices(Pp_vector, 0, buy_vector, 'b', 1, 0, ax1)
    sPp = SMA(Pp_vector, 0, N3)
    ax1.plot(sPp.iloc[N3:], 'purple')

    pd_Pp_dx = pddx(sPp, N4)
    #new = pd.concat([bid, ask, pd_sdata_dx, pd_Pp_dx], axis=1)
    #new.columns = ['bid', 'ask', 'signal1', 'signal2']
    #new['signal2'] = -1 * new['signal2']
    #print(new)
    #Pptn, bn, n, kn, Pp_vectorn = pp2(new, thrs)
    #print(f'The Profit would be {Pptn - k * F}')
    ax2 = ax1.twinx()
    ax2.plot(pd_Pp_dx.iloc[N3:], 'red')
    ax2.axhline(0)
    plt.title('Profit vs Index')
    plt.show()
if plot_SMA_and_dx:
    fig, bx1 = plt.subplots()

    bx1.plot(sdata.iloc[N1:], color = 'black')
    # bx1.plot(nbid, color = 'blue')
    plot_values_at_indices(sdata, 'Extracted Values', sell_vector, 'r', 1, 0, bx1)
    plot_values_at_indices(sdata, 'Extracted Values', buy_vector, 'b', 1, 0, bx1)

    bx2 = bx1.twinx()
    bx2.axhline(0, color = 'red')
    bx2.plot(pd_sdata_dx.iloc[N1 + N2:], color = 'purple')
    #bx2.plot(p2d_sdata_d2x.iloc[N1 + 2 * N2:], color='pink')
    plt.show()
if dx_vs_dx2:
    plt.plot(pd_sdata_dx.iloc[N1 + N2:], color='purple')
    #plt.plot(p2d_sdata_d2x.iloc[N1 + 2 * N2:], color='pink')
    plt.axhline(0, color = 'black')
    plt.show()