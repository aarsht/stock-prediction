import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import lime
import lime.lime_tabular
from keras.models import load_model
import shap

time_step = 2
data_shape = 11
train_ind = 3500
test_ind = 3705
reframed = pd.read_csv('Label_data.csv', header=0, parse_dates=[0])
encoded = pd.read_csv('tech_ind_selected.csv', header=0, parse_dates=[0],index_col=0)
scaler = MinMaxScaler()

train = encoded.iloc[0:train_ind,:]
tr_lime = train
for i in range(0,encoded.shape[1]):
    train.iloc[:,i] = scaler.fit_transform(train.iloc[:,i].values.reshape(-1,1))
    file_str = "scaler_%s.pkl" % (encoded.columns.values[i])
    joblib.dump(scaler,filename=file_str)
train = train.values
print(train)
X_train = []
y_train = []
for i in range(time_step,train_ind):
    for j in range(0, data_shape):
        X_train.append(train[i-time_step:i,j])

y_train = reframed['Label']
y_train = y_train.iloc[time_step:train_ind]
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (int(X_train.shape[0]/data_shape),data_shape,X_train.shape[1]))
X_train = np.transpose(X_train,(0,2,1))
print(X_train.shape)

test = encoded.iloc[train_ind-time_step:test_ind,:]
for i in range(0,encoded.shape[1]):
    file_str = "scaler_%s.pkl" % (encoded.columns.values[i])
    scaler = joblib.load(file_str)
    test.iloc[:,i] = scaler.fit_transform(test.iloc[:,i].values.reshape(-1,1))

test = test.values
X_test = []
y_test = []
for i in range(time_step,test.shape[0]+1):
    for j in range(0, data_shape):
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

regressor.add(LSTM(units=80, return_sequences=True, input_shape=(X_train.shape[1],data_shape)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=80, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=80, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=80))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['acc','mse'])
# Fitting to the training set
regressor.fit(X_train,y_train,epochs=70,batch_size=32,validation_data=(X_test,y_test),verbose=1)
regressor.save('lstm_stock.h5')

regressor = load_model('lstm_stock.h5')
pred_x = regressor.predict_classes(X_train)
random_ind = np.random.choice(X_train.shape[0], 1000, replace=False)
print(random_ind)
data = X_train[random_ind[0:500]]
e = shap.DeepExplainer((regressor.layers[0].input, regressor.layers[-1].output),data)
test1 = X_train[random_ind[500:1000]]
shap_val = e.shap_values(test1)
joblib.dump(shap_val,filename="shapval3.pkl")