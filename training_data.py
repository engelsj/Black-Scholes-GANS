import numpy as np
import yfinance as yf
import math

msft = yf.Ticker("MSFT")
data = msft.history(period="1y")['Close']
def get_y(x):
    return 10 + x*x*x


def sample_data(n=10000, scale=100):
    '''
    data = []

    x = scale*(np.random.random_sample((n,))-0.5)

    for i in range(n):
        yi = get_y(x[i])
        data.append([x[i], yi])
    '''
    return np.array(list(zip(list(range(0,252)), data.values)))