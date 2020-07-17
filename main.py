import math
import numpy as np
import numpy.random as npr
from pylab import plt, mpl
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
npr.seed(100)
np.set_printoptions(precision=4)


def generate_var_reduced_random(partitions, trials):
    sn = npr.standard_normal((partitions + 1, trials))
    return (sn - sn.mean()) / sn.std()


def american_option_ols_monte_carlo_valuation(trials, partitions, time, S0, r, sigma, strike, option):
    dt = time/partitions
    df = np.exp(-r * dt)
    # Simulate underlying
    S = np.zeros((partitions + 1, trials))
    S[0] = S0
    random_mesh = generate_var_reduced_random(partitions, trials)
    for t in range(1, partitions + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * random_mesh[t])
    # Payoff by option type - vectorized because option has a continuation or 0 value at each time step
    if option == 'call':
        h = np.maximum(S - strike, 0)
    if option == 'put':
        h = np.maximum(strike - S, 0)
    # Least-Squares Monte Carlo
    V = np.copy(h)
    for t in range(partitions -1, 0, -1): # 49, 48, … 1)
        '''
        Generate weights a sixth degree polynomial on x= underlying price y = risk-neutral option price 
        at next time step
        '''
        reg = np.polyfit(S[t], V[t + 1] * df, 5)
        C = np.polyval(reg, S[t]) # The continuation value is the evaluation of the OLS regression at time t
        '''
        If the continuation value is greater than the option value, take the continuation value of the next step.
        Otherwise take the option value.
        '''
        V[t] = np.where(C > h[t], V[t +1] * df, h[t])
    # Monte Carlo mean on risk-neutral option value of pathways
    C0 = df * np.mean(V[1])
    return C0, C, V, h # MC Mean, Continuation value paths, Option value paths, Option end distribution


def generate_american_price_surface(strikes, option):
    price_mean = []
    for K in strikes:
        mean_strike, continuation_value_paths, option_value_paths, option_distribution  =  american_option_ols_monte_carlo_valuation(
            trials=100,
            partitions=50,
            time=1,
            S0=100,
            r=0.05,
            sigma=0.1,
            strike=K,
            option=option)
        price_mean.append(mean_strike)
    return price_mean


def main():
    strike_list = np.arange(80., 140.1, 5.)
    put_surface = generate_american_price_surface(strike_list, 'put')
    call_surface = generate_american_price_surface(strike_list, 'call')
    straddle_surface = np.add(put_surface, call_surface)
    plt.figure(figsize=(20, 12))
    plt.plot(strike_list, straddle_surface, 'bo')
    plt.title('Straddle Price by Strike')
    plt.ylabel('Straddle Price')
    plt.xlabel('Strike')
    plt.show()


if __name__ == "__main__":
    main()

