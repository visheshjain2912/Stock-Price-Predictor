#import packages
import pandas as pd
import numpy as np
from datetime import datetime
from pandas_datareader import data as web

#to plot the data
import matplotlib.pyplot as plt
plt.style.use('bmh')

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#fetching and storing the data in a dataframe
startingDate = '2013-01-01'
today = datetime.today().strftime('%Y-%m-%d')
df = pd.DataFrame()
df = web.DataReader('AAPL', data_source = 'yahoo', start = startingDate, end = today)

#print the head
print(df.head())

# Get the number of trading days
print(len(df))

# Visualise the close price data
plt.figure(figsize=(20, 12))
plt.title('Close Price History')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(df['Close'])
plt.show()

#setting index as date values
df['Date'] = pd.to_datetime(df.index.values, format = '%d-%m-%Y')
df.index = df['Date']

#sorting
data = df.sort_index(ascending=True, axis=0)
print(data)

#creating a separate dataset
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

print(new_data.head())

from fastai.tabular import  add_datepart
add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

print(new_data.head())

new_data['mon/fri'] = 0
for i in range(0,len(new_data)):
    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
        new_data['mon/fri'][i] = 1
    else:
        new_data['mon/fri'][i] = 0
print(new_data.head())

#split into train and validation/test
train = new_data[:int(len(new_data)/2)]
valid = new_data[int(len(new_data)/2):]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

#implement linear regression
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(x_train,y_train)

#make predictions and find the rmse
preds1 = model1.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds1)),2)))
print(rms)

# plot
valid['Predictions'] = preds1

valid.index = new_data[int(len(new_data)/2):].index
train.index = new_data[:int(len(new_data)/2)].index

plt.figure(figsize=(20, 12))
plt.title('Linear Regression Model')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Orig', 'Valid', 'Pred'])
plt.show()

# implement k-NN Regression
from sklearn.neighbors import KNeighborsRegressor
model2 = KNeighborsRegressor()
model2.fit(x_train,y_train)

#make predictions and find the rmse
preds2 = model2.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds2)),2)))
print(rms)

# plot
valid['Predictions'] = preds2

valid.index = new_data[int(len(new_data)/2):].index
train.index = new_data[:int(len(new_data)/2)].index

plt.figure(figsize=(20, 12))
plt.title('KNeighbors Regression Model')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Orig', 'Valid', 'Pred'])
plt.show()

# implement Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
model3 = DecisionTreeRegressor()
model3.fit(x_train,y_train)

#make predictions and find the rmse
preds3 = model3.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds3)),2)))
print(rms)

# plot
valid['Predictions'] = preds3

valid.index = new_data[int(len(new_data)/2):].index
train.index = new_data[:int(len(new_data)/2)].index

plt.figure(figsize=(20, 12))
plt.title('Decision Tree Regression Model')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Orig', 'Valid', 'Pred'])
plt.show()

# implement Suppor Vector Regression
from sklearn.svm import SVR
model4 = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)
model4.fit(x_train, y_train)

#make predictions and find the rmse
preds4 = model4.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds4)),2)))
print(rms)

# plot
valid['Predictions'] = preds4

valid.index = new_data[int(len(new_data)/2):].index
train.index = new_data[:int(len(new_data)/2)].index

plt.figure(figsize=(20, 12))
plt.title('SVM Regression Model')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Orig', 'Valid', 'Pred'])
plt.show()

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[:int(len(new_data)/2),:]
valid = dataset[int(len(new_data)/2):,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model5 = Sequential()
model5.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model5.add(LSTM(units=50))
model5.add(Dense(1))

model5.compile(loss='mean_squared_error', optimizer='adam')
model5.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
preds5 = model5.predict(X_test)
preds5 = scaler.inverse_transform(preds5)

rms=np.sqrt(np.mean(np.power((valid-preds5),2)))
print(rms)

# plot

valid = new_data[int(len(new_data)/2):]
train = new_data[:int(len(new_data)/2)]

valid['Predictions'] = preds5

plt.figure(figsize=(20, 12))
plt.title('LSTM Model')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Orig', 'Valid', 'Pred'])
plt.show()