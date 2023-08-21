import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def moving_average(arr, n):

    """Compute the moving average for a given array."""

    arr = np.array(arr)

    if n == 1:

        return np.mean(arr)

    else:

        return np.mean(arr[-n:]) + (np.mean(arr[:-n]) - np.mean(arr)) / (n - 1)

def main():

    # Load example data

    df = pd.read_csv('asset_data.csv')

    prices = df['Open'].values

    # Scale the data using MinMaxScaler

    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices).reshape(-1, 1)
    

    # Compute different moving averages

    sma_list = [moving_average(scaled_prices, i) for i in range(3)]

    # Plot the results

    fig, ax = plt.subplots()

    ax.plot(scaled_prices, label='Original Price')

    ax.axvline(x=0, color='red', linestyle='--', label='Time')

    for i in range(len(sma_list)):

        ax.plot(sma_list[i], label='SMA{}'.format(i+2), color='blue')

    ax.legend()

    ax.set_xlabel('Time')

    ax.set_ylabel('Price')

    plt.show()

if __name__ == '__main__':

    main()

