import math
import numpy as np
import numpy.random as npr
import pandas as pd
import yfinance as yf
from pylab import plt, mpl

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
npr.seed(100)
np.set_printoptions(precision=4)


def generate_var_reduced_random(partitions, trials):
    sn = npr.standard_normal((partitions + 1, trials))
    return (sn - sn.mean()) / sn.std()


def european_monte_carlo_valuation(trials, partitions, time, S0, r, sigma, strike, option):
    dt = time / partitions
    S = np.zeros((partitions + 1, trials))
    # Add more here
    S[0] = S0
    random_mesh = generate_var_reduced_random(partitions, trials)
    for t in range(1, partitions + 1):
        S[t] = S[t - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * random_mesh[t])  # Run stochastic process
    if option == 'call':
        hT = np.maximum(S[-1] - strike, 0)
    if option == 'put':
        hT = np.maximum(strike - S[-1], 0)
    C0 = math.exp(-r * time) * np.mean(hT)  # Risk neutral mean of trials
    return S, hT, C0


def main():
    msft = yf.Ticker("MSFT")
    euro_security_process, euro_trials, euro_price_mean = european_monte_carlo_valuation(trials=10000, partitions=50,
                                                                                         time=1.0, S0=msft.info.get("previousClose"), r=0.05,
                                                                                         sigma=0.64, strike=210, option='call')

    simulated_returns_data = pd.DataFrame(euro_security_process[:, :200])
    simulated_returns = np.log(1 + simulated_returns_data.astype(float).pct_change())

    actual_returns_data = msft.history(period="1y")['Close']
    actual_returns_data = np.log(1 + actual_returns_data.astype(float).pct_change())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    ax1.hist(simulated_returns.values.flatten(), bins=50)
    ax1.set_xlabel('Theoretical Returns')
    ax1.set_ylabel('Frequency')
    ax2.hist(actual_returns_data, bins=50)
    ax2.set_xlabel('Observed Returns')
    ax2.set_ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    main()
