import numpy as np
import pprint
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

time_step = 2
data_shape = 11
train_ind = 3500
test_ind = 3705
reframed = pd.read_csv('Label_data.csv', header=0, parse_dates=[0])
encoded = pd.read_csv('stock_selected.csv', header=0, parse_dates=[0])

train = encoded.values
train = train[0:train_ind,:]
X_train = []
y_train = []
for i in range(time_step,train_ind):
    for j in range(1, data_shape+1):
        X_train.append(train[i-time_step:i,j])

y_train = reframed['Label']
y_train = y_train.iloc[time_step:train_ind]
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping X_train for efficient modelling
X_train = np.reshape(X_train, (int(X_train.shape[0]/data_shape),data_shape,X_train.shape[1]))
X_train = np.transpose(X_train,(0,2,1))
print(X_train.shape)
# print(X_train)
# np.set_printoptions(threshold=np.inf)
# print(y_train)

test = encoded.values
test = test[train_ind-time_step:test_ind,:]
X_test = []
y_test = []
for i in range(time_step,test.shape[0]+1):
    for j in range(1, data_shape+1):
        X_test.append(test[i-time_step:i,j])
print(X_test)
y_test = reframed['Label']
y_test = y_test.iloc[train_ind:test_ind+1]
X_test, y_test = np.array(X_test), np.array(y_test)
# Reshaping X_train for efficient modelling
X_test = np.reshape(X_test, (int(X_test.shape[0]/data_shape),data_shape,X_test.shape[1]))
X_test = np.transpose(X_test,(0,2,1))
print(X_test)
np.set_printoptions(threshold=np.inf)
print(y_test)
# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=80, return_sequences=True, input_shape=(X_train.shape[1],data_shape)))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=80, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=80, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=80))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['acc','mse'])
# Fitting to the training set
regressor.fit(X_train,y_train,epochs=70,batch_size=32,validation_data=(X_test,y_test),verbose=1)
regressor.save('lstm_stock.h5')