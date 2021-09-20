# %% importing libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas._libs.tslibs import timestamps
import statsmodels.api as sm
from pylab import rcParams
import itertools
import warnings
from statsmodels.discrete.discrete_model import L1NegativeBinomialResults
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings('ignore')
# %% setting parameters for plotting
plt.style.use('fivethirtyeight')
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['text.color'] = 'k'
rcParams['figure.figsize'] = 18, 8
# %% loading training data
path = 'data/train.csv'
data = pd.read_csv(path)
data.drop(columns='ID', axis=1, inplace=True)
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d-%m-%Y %H:%M')
data.isnull().sum()
data.set_index('Datetime', inplace=True)
data.sort_index(inplace=True)
# %% examining seasonality for hourly data
data_hourly = data
decomposition = sm.tsa.seasonal_decompose(
    x=data['Count']['2013-01'], model='additive', period=24)
fig = decomposition.plot()
plt.show()
# %% Autocorrelation for hourly data
sm.graphics.tsa.plot_acf(x=data_hourly['Count']['2013-1'], lags=60)
plt.show()
# %% examining seasonality for daily data
data_daily = data.resample('1d').mean()
decomposition_daily = sm.tsa.seasonal_decompose(
    x=data_daily['Count']['2013'], model='additive', period=7
)
fig = decomposition_daily.plot()
plt.show()
# %% Autcorellation for daily
sm.graphics.tsa.plot_acf(x=data_daily['Count']['2013'], lags=20)
plt.show()
# %% Running ARIMA to predict future: setting parameters for daily data
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
S_PDQ = [(x[0], x[1], x[2], 7) for x in pdq]
# %% Parameter tuning for daily data
results_list = []
for param in pdq:
    for s_param in S_PDQ:
        try:
            model = sm.tsa.SARIMAX(data_daily['Count'],
                                   order=param,
                                   seasonal_order=s_param,
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)
            result = model.fit()
            results_list.append(
                ('AIC for ARIMA{0}x{1}'.format(param, s_param), result.aic))
        except:
            print('There is an error with parameters')
# %% best 5 results from above grid search
best_5results = [('AIC for ARIMA(1, 1, 2)x(0, 2, 2, 7)', 7047.594445231808),
                 ('AIC for ARIMA(0, 1, 2)x(0, 2, 2, 7)', 7053.530767955017),
                 ('AIC for ARIMA(0, 1, 2)x(1, 2, 2, 7)', 7055.467590677273),
                 ('AIC for ARIMA(1, 1, 1)x(0, 2, 2, 7)', 7056.132810524319),
                 ('AIC for ARIMA(2, 1, 1)x(1, 2, 2, 7)', 7059.284389950708)]
# %% running best result
best_model = sm.tsa.SARIMAX(data_daily['Count'],
                            order=(1, 1, 2),
                            seasonal_order=(0, 2, 2, 7),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
result = best_model.fit()
print(result.summary().tables[1])
# %% plotting diagnostics
result.plot_diagnostics()
plt.show()
# %% prediction
y_pred = result.get_prediction(start=pd.to_datetime('2014-03-01'),
                               dynamic=False)
y_pred_ci = y_pred.conf_int()

ax = data_daily['Count']['2014':].plot(label='observed')
y_pred.predicted_mean.plot(ax=ax, label='one step ahead forecast',
                           alpha=0.7)
ax.fill_between(y_pred_ci.index,
                y_pred_ci.iloc[:, 0],
                y_pred_ci.iloc[:, 1], color='k', alpha=0.2)
ax.set_xlabel('Date')
ax.set_ylabel('Count per hour')
plt.legend()
plt.show()
# %% prediction metrics
y_forecasted = y_pred.predicted_mean
y_truth = data_daily['Count']['2014-03-01':]
rmse = np.sqrt(((y_truth-y_forecasted)**2).mean())
print('RMSE for the model is {}'.format(round(rmse)))
# %% loading test data
path = 'data/test.csv'
data_test = pd.read_csv(path)
data_test['Datetime'] = pd.to_datetime(
    data_test['Datetime'], format='%d-%m-%Y %H:%M')
data_test.set_index('Datetime', inplace=True)
data_test.sort_index(inplace=True)

# %% making test data in daily form
data_test_daily = data_test.resample('1d').mean()
data_test_daily.shape
# %% capturing daily seasonal feature of traffic data
decomposition_daily = sm.tsa.seasonal_decompose(
    data_hourly['2014-8'], period=24)
decomposition_daily.plot()
plt.show()
seasonal_component = decomposition_daily.seasonal['2014-08-01']
# %% prediction for test data
y_future = result.get_forecast(steps=213)
y_future_ci = y_future.conf_int()
ax = data_daily['Count']['2014-06-01':].plot(label='observed')
y_future.predicted_mean.plot(ax=ax, label='future')
ax.fill_between(y_future_ci.index,
                y_future_ci.iloc[:, 0],
                y_future_ci.iloc[:, 1], color='k', alpha=0.2
                )
ax.set_xlabel('Date')
ax.set_ylabel('Cout per hour')
plt.legend()
plt.show()
# %% predicting testing data with daily predicted feature
data_test['Count'] = 0
for i in range(len(data_test)):
    data_test.Count[i] = y_future.predicted_mean.loc[str(data_test.index[i].date())] +\
        seasonal_component.loc['2014-08-01'+' '+str(data_test.index[i].time)]
# %% adding seasonal feature for one day to the predicted results
seasonal_component = pd.DataFrame(seasonal_component)
for i in range(len(data_test)):
    data_test.Count[i] += seasonal_component.loc['2014-08-01' +
                                                 ' '+str(data_test.index[i].time())]

# %% prepating file for submission
data_test.to_csv('data/submission.csv', index=False)
