import quandl
import pandas as pd
from googlefinance.client import get_price_data
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def get_data_google(symbol, years, exchange):
    param = {
        'q': symbol,  # Stock symbol (ex: "AAPL")
        'i': "86400",  # Interval size in seconds ("86400" = 1 day intervals)
        'x': exchange,  # Stock exchange symbol on which stock is traded (ex: "NASD")
        'p': years
    }
    data = get_price_data(param)

    return data


def get_data_quandl(symbol, start_date, end_date):
    """
    :param symbol: ETF Symbol for retreival from quandl
    :param start_date: Start date of data
    :param end_date: End date of data
    :return: Stock performance for requested period
    """
    auth_key = '7VBcsWq-7Fyr3ByV-hQq'
    data = quandl.get(symbol, start_date=start_date, end_date=end_date)
    # data = quandl.get(dataset=)
    # data = quandl.get()
    return data

def generate_features(df):
    """
    Generate features for a stock/index based on historical prices
    :param df: pandas dataframe file
    :return: dataset with new features
    """
    df_new = pd.DataFrame()

    # Six original features
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    # Shift index by one in order to take the value of previous
    # For example, [1,3,4,5] -> [NA, 1, 3, 4]

    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)

    # 31 Original features
    # average price
    df_new['avg_price_5'] = pd.rolling_mean(df['Close'],window=5).shift(1)

    # rolling_mean claculates the moving average given a window. For example [1, 2, 1, 4, 3, 2, 1, 4]
    # -> [N/A, N/A, N/A, N/A, 2.2, 2.4, 2.2, 2.8]
    df_new['avg_price_30'] = pd.rolling_mean(df['Close'], window=21).shift(1)
    df_new['avg_price_365'] = pd.rolling_mean(df['Close'], window=252).shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']

    # average volume
    df_new['avg_volume_5'] = pd.rolling_mean(df['Volume'], window=5).shift(1)
    df_new['avg_volume_30'] = pd.rolling_mean(df['Volume'], window=21).shift(1)
    df_new['avg_volume_365'] = pd.rolling_mean(df['Volume'], window=252).shift(1)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']

    # Standard deviation of prices
    df_new['std_price_5'] = pd.rolling_std(df['Close'], window=5).shift(1)

    # rolling_mean calculates the moving standard deviation given a window
    df_new['std_price_30'] = pd.rolling_std(df['Close'], window=21).shift(1)
    df_new['std_price_365'] = pd.rolling_std(df['Close'], window=252).shift(1)
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']

    # Standard deviation of volumes
    df_new['std_volume_5'] = pd.rolling_std(df['Volume'], window=5).shift(1)
    df_new['std_volume_30'] = pd.rolling_std(df['Volume'], window=21).shift(1)
    df_new['std_volume_365'] = pd.rolling_std(df['Volume'], window=252).shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']

    # Stock return
    df_new['return_1'] = ((df['Close']-df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['Close']-df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['Close']-df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)

    df_new['moving_avg_5'] = pd.rolling_mean(df_new['return_1'], window=5)
    df_new['moving_avg_30'] = pd.rolling_mean(df_new['return_1'], window=21)
    df_new['moving_avg_365'] = pd.rolling_mean(df_new['return_1'], window=252)

    # The target
    df_new['close'] = df['Close']
    print('---DF NEW--')
    print('---DF NEW--')
    print('---DF NEW--')
    print(df_new.tail(5))
    print('---DF NEW--')
    print('---DF NEW--')
    print('---DF NEW--')
    df_new = df_new.dropna(axis=0)  # This will drop rows with any N/A value, which is by-product of moving average/std.
    return df_new

# data = get_data_quandl('TVC/DJI', '2001-01-01', '2015-12-31')

# data = get_data_google('F', '17Y','NYSE')
data = get_data_google('AMD', '17Y','NASDAQ')
today = datetime.datetime(2017, 9, 1, 16, 0)

data.loc[today] = [13.15, None, None, 13, None]

print(data)

data = generate_features(data)

# Start End Dates for training data
start_train = datetime.datetime(2001, 1, 1, 16, 00)
end_train= datetime.datetime(2017, 5, 1, 16, 00)
data_train = data.ix[start_train:end_train]

# print('===================')
# print(data_train.round(decimals=3).head(3))
# print(data_train.round(decimals=3).tail(3))
# print('===================')

X_columns = list(data.drop(['close'], axis=1).columns)
y_column = 'close'
X_train = data_train[X_columns]
y_train = data_train[y_column]

print(X_train.shape)
print(y_train.shape)

# Start End Dates for test data
start_test = datetime.datetime(2017, 5, 1, 16, 0)
end_test = datetime.datetime(2017, 12, 31, 16, 0)
data_test = data.ix[start_test:end_test]
today = datetime.datetime(2017, 8, 30, 16, 0)

print('==Data TESTTTT')

# print(data_test.columns)

print(data_test)

X_test = data_test[X_columns]
y_test = data_test[y_column]

print('===== SGD Results====')
# print(data_test)

print(len(y_test.values))
print(len(y_test.index))
# print(y_test)
# print(sgd_predictions)

x_plot = list(y_test.index)
y_plot = list(y_test.values)

print('===== SGD Resuts ====')
print(X_test.shape)
print(y_test.shape)


# Start regression ML with standard Scaler
scaler = StandardScaler()

scaler.fit(X_train)

X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)



param_grid = {
   "alpha": [3e-06, 1e-5, 3e-5],
   "eta0": [0.01, 0.03, 0.1],
   }

lr = SGDRegressor(penalty='l2', max_iter=1000)
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_scaled_train, y_train)

print('+++++++++++')
print(grid_search.best_params_)
print('+++++++++++')

lr_best = grid_search.best_estimator_
sgd_predictions = lr_best.predict(X_scaled_test)



print('===== SGD ====')
print(len(sgd_predictions))
print('MSE: {0:.3f}'.format(mean_squared_error(y_test, sgd_predictions)))
print('MAE: {0:.3f}'.format(mean_absolute_error(y_test, sgd_predictions)))
print('R^2: {0:.3f}'.format(r2_score(y_test, sgd_predictions)))

print('====PREDICTIONS=======')
print('SGD')
print(sgd_predictions)

# print('====PREDICTIONS=======')
#
# # Random Forest
# param_grid = {
#    "max_depth": [30, 50],
#    "min_samples_split": [3, 5, 10],
#    }
# rf = RandomForestRegressor(n_estimators=1000)
# grid_search = GridSearchCV(rf, param_grid, cv=5,
#                                  scoring='neg_mean_absolute_error')
# grid_search.fit(X_train, y_train)
#
# print(grid_search.best_params_)
# rf_best = grid_search.best_estimator_
# rf_predictions = rf_best.predict(X_test)
# print('=====Random Forest====')
# print('MSE: {0:.3f}'.format(mean_squared_error(y_test, rf_predictions)))
# print('MAE: {0:.3f}'.format(mean_absolute_error(y_test, rf_predictions)))
# print('R^2: {0:.3f}'.format(r2_score(y_test, rf_predictions)))
#
# print('====PREDICTIONS=======')
# print('SGD')
# print(sgd_predictions)
# print('RF')
# print(rf_predictions)
#
# print('====PREDICTIONS=======')
#
# # SVR Prediction
#
# param_grid = {
#             "C": [1000, 3000, 10000],
#             "epsilon": [0.00001, 0.00003, 0.0001],
#             }
# print('svr 1')
# svr = SVR(kernel='linear')
# print('svr grid_search')
# grid_search = GridSearchCV(svr, param_grid, cv=5,
#                                   scoring='neg_mean_absolute_error')
# print('svr grid search fit')
# grid_search.fit(X_scaled_train, y_train)
# print(grid_search.best_params_)
# print('svr best')
# svr_best = grid_search.best_estimator_
# print('svr prediction')
# svr_predictions = svr_best.predict(X_scaled_test)
#
# print('MSE: {0:.3f}'.format(mean_squared_error(y_test, svr_predictions)))
# print('MAE: {0:.3f}'.format(mean_absolute_error(y_test, svr_predictions)))
# print('R^2: {0:.3f}'.format(r2_score(y_test, svr_predictions)))
#
# print('====PREDICTIONS=======')
# print('SGD')
# print(sgd_predictions)
# print('RF')
# print(rf_predictions)
# print('SVR')
# print(svr_predictions)
#
# print('====PREDICTIONS=======')

plt.plot(x=y_test.index, y=y_test.values, c='k')
plt.scatter(x=y_test.index, y=y_test.values, marker='o', c='k')
plt.scatter(x=y_test.index, y=sgd_predictions, marker='*', c='b')
# plt.scatter(x=y_test.index, y=rf_predictions, marker='r', c='r')
# plt.scatter(x=y_test.index, y=svr_predictions, marker='s', c='y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
