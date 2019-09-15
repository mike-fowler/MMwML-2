# Import dependancies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Import csv data file as a pandas dataframe
df = pd.read_csv('nvda.csv')

# Create the Lists / X and y data set
dates = []
prices = []
dates_train = []
prices_train = []
dates_test = []
prices_test = []

### Actual stock price data for month
df_dates = df.loc[:,'Date'] # The Date column of the dataset is stored into a variabe df_dates
df_close = df.loc[:,'Adj Close'] # The Adj Close column of the dataset is stored into a variabe df_close

# Make List of days from Date column in DataFrame
for date in df_dates:
    dates.append([int(date.split('-')[2])])

# Make List of Adjusted Closing Prices
for close in df_close:
    prices.append(float(close))

### Create Training Data set by removing last 5 rows from DataFrame
df_train = df.head(len(df)-5)

df_dates_train = df_train.loc[:,'Date'] # The date column of the dataset is stored into a variabe df_dates_train
df_close_train = df_train.loc[:,'Adj Close'] # The open column of the dataset is stored into a variabe df_close_train

# Make List of days from date column in DataFrame for training
for date in df_dates_train:
    dates_train.append([int(date.split('-')[2])])

# Make list of Adjusted Closing Prices for training
for close in df_close_train:
    prices_train.append(float(close))

### Create Test Data set from last 5 rows from DataFrame
df_test = df.tail(6)

df_dates_test = df_test.loc[:,'Date'] # The date column of the dataset is stored into a variabe df_dates_test
df_close_test = df_test.loc[:,'Adj Close'] # The open column of the dataset is stored into a variabe df_close_test

# Make list of Days from date column in DataFrame for testing
for date in df_dates_test:
    dates_test.append([int(date.split('-')[2])]) # Reduced to days, month and year removed

# Make list of Adjusted Closing Prices for testing
for close in df_close_test:
    prices_test.append(float(close))

### Input regression models
# Linear regression
lin_reg = LinearRegression()
#Train model on training dataset
lin_reg.fit(dates_train, prices_train)

# SVR - Polynomial
svr_poly= SVR(kernel='poly', C=1e3, degree=2)
#Train model on training dataset
svr_poly.fit(dates_train, prices_train)

# SVR - RBF
svr_rbf = SVR(kernel='rbf', C=1e3, gamma='scale')
#Train model on training dataset
svr_rbf.fit(dates_train, prices_train)

### Plot data
plt.plot(dates, prices, color='green', label='Actual Data', linewidth=3.0)
plt.scatter(dates, prices, color='green')

plt.plot(dates_train, lin_reg.predict(dates_train), color='red', label='Linear Reg', linewidth=2.0)
plt.plot(dates_test, lin_reg.predict(dates_test), color='red', linewidth=2.0, linestyle='dashed')

plt.plot(dates_train, svr_poly.predict(dates_train), color='orange', label='SVR Poly', linewidth=2.0)
plt.plot(dates_test, svr_poly.predict(dates_test), color='orange', linewidth=2.0, linestyle='dashed')

plt.plot(dates_train, svr_rbf.predict(dates_train), color='blue', label='SVR RBF', linewidth=2.0)
plt.plot(dates_test, svr_rbf.predict(dates_test), color='blue', linewidth=2.0, linestyle='dashed')

plt.axvline (x=23, linewidth=1.0, color='gray')
plt.xlabel('Day of Month')
plt.ylabel('Adjusted Closing Price')
plt.title('Stock Price Prediction for NVIDIA in August 2019')
plt.legend()
plt.show()

### Predictions at day 30 (end of month)
x=[[30],[0]] # Day 30

print ("")
print ("Price predictions made for the end of the month on the 23rd of August")
print ("----------------------------------------------------------------------")
print (f'The Actual price at the end of August:    ${prices[len(prices)-1]:.4f}')
print ("")
print (f'The svr_rbf predicted price:              ${svr_rbf.predict(x)[0]:.4f}')
print ("")
print (f'The svr_poly predicted price:             ${svr_poly.predict(x)[0]:.4f}')
print ("")
print (f'The lin_reg predicted price:              ${lin_reg.predict(x)[0]:.4f}')
print ("----------------------------------------------------------------------")
