from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


time_step = 2
data_shape = 11
train_ind = 3500
test_ind = 3705
reframed = pd.read_csv('Label_data.csv', header=0, parse_dates=[0])
encoded = pd.read_csv('tech_ind_selected.csv', header=0, parse_dates=[0],index_col=0)
bse_data = pd.read_csv('BSESN.csv', header=0, parse_dates=[0],index_col=0)
scaler = MinMaxScaler()

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
regressor = load_model('lstm_stock.h5')
predicted_percentage = regressor.predict(X_test)
print(y_test)
prediction = pd.Series()
bi_pred = [1 if x>0.50 else 0 for x in predicted_percentage]
print(bi_pred)
compare = [1 if x==y else 0 for x,y in zip(y_test,bi_pred)]
print(compare)
# colors = {0:'red', 1:'blue'}
# correct = np.ma.masked_where(compare == 0,compare)
# incorrect = np.ma.masked_where(compare == 1,compare)
# fig,ax = plt.subplots()
# bse_close = bse_data['Adj Close']
# ax.plot = (bse_close.iloc[train_ind:test_ind+1], correct, bse_close.iloc[train_ind:test_ind+1], incorrect)
bse_close = pd.DataFrame()
bse_close = bse_data['Adj Close']
bse_close = bse_close.iloc[train_ind:test_ind+1]
bse_close['label'] = pd.Series(compare)
colors = {0:'red', 1:'blue'}
fig,ax = plt.subplots()
group = np.array([1,0])
for g in group:
        ix = np.where(bse_close['label'] == g)
        print(ix)
        plt.plot(bse_close.iloc[ix],'.',c=colors[g])
# groups = bse_close.groupby('label')
#
# # Plot
# fig, ax = plt.subplots()
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# for name, group in groups:
#     ax.plot(bse_close['Adj Close'], marker='o', linestyle='', ms=12, label=name)
# ax.legend()
plt.title("True and False predictions")
plt.xlabel("Time-->")
plt.ylabel("Share price")
plt.legend(['True','False'], loc='upper left')
plt.show()