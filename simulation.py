from progressbar import *
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pricing import Black

np.random.seed(0)
plt.rcParams['figure.figsize'] = (16, 4.5)

class BrownianMotion:
    def __init__(self, params):
        self.params = params

    def brownian(x0, n, dt, delta, out=None):
        x0 = np.asarray(x0)
        r = norm.rvs(size=x0.shape + (n,), scale=delta * np.sqrt(dt))
        if out is None:
            out = np.empty(r.shape)

        #Cumulative sum of random samples
        np.cumsum(r, axis=-1, out=out)

        # Add the initial condition.
        out += np.expand_dims(x0, axis=-1)
        return out

#Plot simple Wiener Process
delta = 2
T = 10.0
N = 500
dt = T/N
m = 20 #Num of realizations
# Create an empty array to store the realizations.
x = np.empty((m,N+1))
# Initial values of x.
x[:, 0] = 50
BrownianMotion.brownian(x[:,0], N, dt, delta, out=x[:,1:])

timeframe = np.linspace(0.0, N*dt, N+1)
for k in range(m):
    plt.plot(timeframe, x[k])
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.grid(True)
plt.title('Example Brownian Motion')
plt.show()

dataframe = pd.read_csv('historical_data.csv').set_index('time') / 100  # Convert interest rates to %
pd.options.display.max_rows = 10
print(dataframe)

hist_timeline = list(dataframe.index)
tenors = [eval(x) for x in dataframe.columns] #Get the list of tenors
hist_rates = dataframe.to_numpy()

#Plot historical rates' evolution by tenor
plt.plot(hist_rates)
plt.xlabel(r'Time $t$')
plt.ylabel(r'Historical rate $f(t,\tau)$')
plt.text(200, 0.065, r'Evolution of daily historical yield curve data with 51 tenors over 5 years. Each line represents a different tenor')
plt.title(r'Historical $f(t,\tau)$ by $t$')
plt.show()

#Plot historical rates' evolution by day
plt.plot(tenors, hist_rates.transpose())
plt.xlabel(r'Time $t$')
plt.ylabel(r'Historical rate $f(t,\tau)$')
plt.text(200, 0.065, r'Evolution of daily historical yield curve data with 51 tenors over 5 years. Each line represents a different day')
plt.title(r'Historical $f(t,\tau)$ by $t$')
plt.show()

#Calculate differential of the historical rates with respect to t
diff_rates = np.diff(hist_rates, axis=0)
plt.plot(diff_rates)
plt.xlabel(r'Time $t$')
plt.title(r'Differentiate matrix of historical rates $df(t,\tau)$ by $t$')
plt.show()


# Calculate covariance matrix and its principal component decomposition
sigma = np.cov(diff_rates.transpose())
sigma *= 252 #Get covariace in annual terms
eigval, eigvec = np.linalg.eig(sigma)
eigvec = np.matrix(eigvec)
index_eigvec = list(reversed(eigval.argsort()))[0:3]  # highest principal component first in the array
princ_eigval = np.array([eigval[i] for i in index_eigvec])
princ_comp = np.hstack([eigvec[:, i] for i in index_eigvec])

plt.plot(princ_comp)
plt.title('Eigenvectors of Principal components')
plt.xlabel(r'Time $t$')
plt.show();

sqrt_eigval = np.matrix(princ_eigval ** .5)
tmp_m = np.vstack([sqrt_eigval for i in range(princ_comp.shape[0])])  # resize matrix (1,factors) to (n, factors)
vols = np.multiply(tmp_m, princ_comp)  # multiply matrice element-wise
print('vols shape: ' + str(vols.shape))

plt.plot(vols)
plt.title('Discretized volatilities')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Volatility $\sigma$')
plt.text(8, 0.004, 'sqrt of eigenvalue * eigenvector')
plt.show();


def get_matrix_column(matrix, i):
    return np.array(matrix[:, i].flatten())[0]


class Interpolator:
    def __init__(self, params):
        self.params = params

    def calc(self, x):
        n = len(self.params)
        C = self.params
        X = np.array([x ** i for i in reversed(range(n))])
        return sum(np.multiply(X, C))


fitted_vols = []


def fit_volatility(i, degree, title):
    vol = get_matrix_column(vols, i)
    fitted_vol = Interpolator(np.polyfit(tenors, vol, degree))
    plt.plot(tenors, vol, label='Historical volatility')
    plt.plot(tenors, [fitted_vol.calc(x) for x in tenors], label='Fitted volatility')
    plt.title(title), plt.xlabel(r'Time $t$'), plt.legend()
    fitted_vols.append(fitted_vol)


plt.subplot(1, 3, 1), fit_volatility(0, 0, '1st component');
plt.subplot(1, 3, 2), fit_volatility(1, 3, '2nd component');
plt.subplot(1, 3, 3), fit_volatility(2, 3, '3rd component');
plt.show()


def integrate(f, x0, x1, dx):
    n = (x1 - x0) / dx + 1
    out = 0
    for i, x in enumerate(np.linspace(x0, x1, int(n))):
        if i == 0 or i == n - 1:
            out += 0.5 * f(x)
        else:
            out += f(x)  #Do not adjust by trapezoidal rule
    out *= dx
    return out


mc_tenors = np.linspace(0, 25, 51)
# Discretize fitted volatility functions for monte carlo
mc_vols = np.matrix([[fitted_vol.calc(tenor) for tenor in mc_tenors] for fitted_vol in fitted_vols]).transpose()
plt.plot(mc_tenors, mc_vols)
plt.xlabel(r'Time $t$')
plt.title('Volatilities')
plt.show();


def integration_all(tau, fitted_vols):
    # It uses the fact that volatility is function of time in HJM model
    output = 0.
    for fitted_vol in fitted_vols:
        output += integrate(fitted_vol.calc, 0, tau, 0.01) * fitted_vol.calc(tau)
    return output


#Plot the evolution of drift factor
mc_drift = np.array([integration_all(tau, fitted_vols) for tau in mc_tenors])
plt.plot(mc_drift)
plt.xlabel(r'Time $t$')
plt.title('Risk-neutral drift $(alpha)$')
plt.show();


def simulation(f, tenors, drift, vols, timeline):
    vols = np.array(vols.transpose())  # 3 rows, T columns
    len_tenors = len(tenors)
    len_vols = len(vols)
    yield timeline[0], copy.copy(f)
    for it in range(1, len(timeline)):
        t = timeline[it]
        dt = t - timeline[it - 1]
        sqrt_dt = np.sqrt(dt)
        fprev = f
        f = copy.copy(f)
        random_numbers = [np.random.normal() for i in range(len_vols)]
        for i in range(len_tenors):
            val = fprev + drift[i] * dt
            sum = 0
            for iVol, vol in enumerate(vols):
                sum += vol[i] * random_numbers[iVol]
            val += sum * sqrt_dt
            # take left difference
            iT1 = i + 1 if i < len_tenors - 1 else i - 1
            df_dT = (fprev[iT1] - fprev[i]) / (iT1 - i)
            val += df_dT * dt
            f[i] = val
        yield t, f


proj_rates = []
proj_timeline = np.linspace(0, 5, 500)
spot_rates = np.array(hist_rates[-1,:].flatten())[0]

for i, (t, f) in enumerate(simulation(spot_rates, mc_tenors, mc_drift, mc_vols, proj_timeline)):
    proj_rates.append(f)

#Plot the projected rates
proj_rates = np.matrix(proj_rates)
plt.plot(proj_timeline.transpose(), proj_rates)
plt.xlabel(r'Time')
plt.ylabel(r'Rate');
plt.title(r'Simulated $f(t,\tau)$ by $t$')
plt.show()

plt.plot(mc_tenors, proj_rates.transpose())
plt.xlabel(r'Tenor')
plt.ylabel(r'Rate ');
plt.title(r'Simulated $f(t,\tau)$ by $\tau$')
plt.show();


class Integrator:
    def __init__(self, x0, x1):
        assert x0 < x1
        self.sum, self.n, self.x0, self.x1 = 0, 0, x0, x1

    def add(self, value):
        self.sum += value
        self.n += 1

    def get_integral(self):
        return (self.x1 - self.x0) * self.sum / self.n


t_exp, t_mat = 1., 2.
K = .03
n_simulations, n_timesteps = 500, 50


def mc_simulation(t_exp, t_mat, K, n_simulations, n_timesteps, vol, isCall):
    proj_timeline = np.linspace(0, t_mat, n_timesteps)
    simulated_forecast_rates = []
    simulated_df = []
    simulated_values = []
    for i in range(0, n_simulations):
        rate_forecast = None
        rate_discount = Integrator(0, t_exp)  #Compounded discount
        for t, curve_fwd in simulation(spot_rates, mc_tenors, mc_drift, mc_vols, proj_timeline):
            f_t_0 = np.interp(0., mc_tenors, curve_fwd)  # rate $f_t^0$
            rate_discount.add(f_t_0)
            #t reaches expiration
            if t >= t_exp and rate_forecast is None:
                Tau = t_mat - t_exp
                rate_forecast = Integrator(0, Tau)  # integrate all inst.fwd.rates from 0 till 1Y tenor to get 1Y spot rate
                for s in np.linspace(0, Tau, 15):  # $\int_0^T f(t,s)ds$
                    f_texp_s = np.interp(s, mc_tenors, curve_fwd)
                    rate_forecast.add(f_texp_s)
                rate_forecast = rate_forecast.get_integral()

        df = np.exp(-rate_discount.get_integral())  # Discount factor
        simulated_forecast_rates.append(rate_forecast)
        simulated_df.append(df)

        cap_value = Black.price_cap(rate_forecast, K, 90, vol, 0.02, 90, isCall)
        simulated_values.append(cap_value)

        return simulated_forecast_rates, simulated_df, simulated_values




