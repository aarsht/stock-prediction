import numpy as np
from sklearn.externals import joblib
import shap
from matplotlib import pyplot as plt


shap_val = joblib.load("shapval1.pkl")
shap_val = np.array(shap_val)
shap_val = np.reshape(shap_val,(int(shap_val.shape[1]),int(shap_val.shape[2]),int(shap_val.shape[3])))
shap_abs = np.absolute(shap_val)
print(shap_abs)
# clipval = shap_val.clip(min=0)
print("sum:")
sum1 = np.sum(shap_abs,axis=0)
print(sum1)
f_names = ['RSI_14D','STOK','STOD','ROC','Momentum','CCI','ADX','MACD','Money_Flow_Index','WillR','INRchange']
x_pos = [i for i, _ in enumerate(f_names)]
x_pos = x_pos
print(x_pos)
plt1 = plt.subplot(311)
plt1.barh(x_pos,sum1[1])
plt1.set_yticks(x_pos)
plt1.set_yticklabels(f_names)
plt1.set_title("Yesterday's features (time-step 2)")
plt2 = plt.subplot(312,sharex=plt1)
plt2.barh(x_pos,sum1[0])
plt2.set_yticks(x_pos)
plt2.set_yticklabels(f_names)
plt2.set_title("The day before yesterday's features(time-step 1)")
plt.tight_layout()
plt.show()