import numpy as np
import pandas as pd
import yfinance as yf
from pylab import plt, mpl
from bs4 import BeautifulSoup
import urllib.request
import random

data = pd.read_csv('loaded_stock_prices.csv')
msft = yf.Ticker("MSFT")
msft_data = msft.history(period="1y")['Close']


def sample_data():
    #random_sample = data[random.choice(data.columns)]
    #random_sample = random_sample.dropna()
    #random_sample = np.log(1 + random_sample.astype(float).pct_change())
    #random_sample = random_sample.fillna(0)
    return np.array(list(zip(list(range(0, len(msft_data))), msft_data.values)))


def fetch_price_data_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page, "lxml")
    data_table = soup.find('table', {"id": 'constituents'})
    tickers = []
    for row in data_table.findAll('tr'):
        cells = row.findAll('td')
        if len(cells) == 9:
            tickers.append(cells[0].find(text=True))
    loaded_data = yf.download(" ".join(tickers), period="1y")
    loaded_data = loaded_data.dropna(axis='columns')
    loaded_data = loaded_data.drop('', axis='columns')
    loaded_data.to_csv('loaded_stock_prices.csv')


def display_random_data():
    random_sample = data[random.choice(data.columns)]
    plt.scatter(range(0, len(random_sample)), random_sample.values)
    plt.show()
