import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional, GRU
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import IPython
import keras_tuner as kt
import shutil
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)
from sklearn.ensemble import RandomForestRegressor
# from hyperopt import tpe, hp, Trials
# from hyperopt.fmin import fmin
from sklearn.metrics import mean_squared_error, mean_absolute_error
def get_rmse(y_valid, y_predict):
    return np.sqrt(mean_squared_error(y_valid, y_predict))
import matplotlib.pyplot as plt

df = pd.read_csv('D:\\OneDrive\\3.py_SCH_data\\data_OP_ML\\PM2.5_OP_2019-2021_2.csv')

# 황사일 제거시(To exclude data for cases of yellow dust, please execute the following line)
# df = df[df['황사여부']==0]

df_y = df[['OPv (DTTv)']]

df_x = df[['온도', '습도', '풍속', 'O3', 'NO2', 'CO', 'PM2.5', 'PM10',
       'OC', 'EC', 'WSTN', 'WSIN', 'WSON', 'WSOC', 'Cl', 'NO3', 'SO4',
       'Na', 'NH4', 'K', 'Ca', 'Mg', 'Li', 'Al', 'Ti', 'V', 'Cr',
       'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Cd', 'Pb', 'SO2']]
df_x = df_x.rename(columns={'온도':'Temperature', '습도':'Humidity', '풍속':'Wind speed'})


imputer = KNNImputer(n_neighbors=3) #KNN
df_x.iloc[:,:] = imputer.fit_transform(df_x)

df = pd.concat([df_x,df_y], axis=1)

scalingfactor = {}
data_scaled = df.copy()

for c in df.columns[:]:
    denominator = df[c].max()-df[c].min()
    scalingfactor[c] = [denominator, df[c].min(), df[c].max()]
    data_scaled[c] = (df[c] - df[c].min())/denominator

df = data_scaled.iloc[:, :]

df_y = df[['OPv (DTTv)']]

df_x = df[['Temperature', 'Humidity', 'Wind speed', 'O3', 'NO2', 'CO', 'PM2.5', 'PM10',
       'OC', 'EC', 'WSTN', 'WSIN', 'WSON', 'WSOC', 'Cl', 'NO3', 'SO4',
       'Na', 'NH4', 'K', 'Ca', 'Mg', 'Li', 'Al', 'Ti', 'V', 'Cr',
       'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Cd', 'Pb', 'SO2']]

# 기상자료 제외시
# df_x = df[['O3', 'NO2', 'CO', 'PM2.5', 'PM10',
#        'OC', 'EC', 'WSTN', 'WSIN', 'WSON', 'WSOC', 'Cl', 'NO3', 'SO4',
#        'Na', 'NH4', 'K', 'Ca', 'Mg', 'Li', 'Al', 'Ti', 'V', 'Cr',
#        'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Cd', 'Pb', 'SO2']]

random_state_split= 7

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)#, random_state=random_state_split)

'''
#Kfold 쓴다면(cross validation)
import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=2)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

'''
# 직접 ANN 시작

# create ANN model
model = Sequential()

# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=10, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(1, activation='relu', kernel_initializer='normal'))
# Compiling the model
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=1e-3))

# Fitting the ANN to the Training set
history = model.fit(np.array(x_train), np.array(y_train), batch_size=20, epochs=100, verbose=1)

# 직접 ANN 끝

y_predicted = model.predict(x_test)
evaluation = model.evaluate(x_test, y_test)

y_pred_train = model.predict(x_train.values)
y_pred_valid = model.predict(x_test.values)
print("Y Pred Train Shape : ", y_pred_train.shape)
print("Y Pred Test  Shape : ", y_pred_valid.shape)


# Denormalize the data
y_pred_train = y_pred_train * scalingfactor['OPv (DTTv)'][0] + scalingfactor['OPv (DTTv)'][1]
y_train = y_train * scalingfactor['OPv (DTTv)'][0] + scalingfactor['OPv (DTTv)'][1]
y_test = y_test * scalingfactor['OPv (DTTv)'][0] + scalingfactor['OPv (DTTv)'][1]
y_pred_valid = y_pred_valid * scalingfactor['OPv (DTTv)'][0] + scalingfactor['OPv (DTTv)'][1]


train_mse = mean_squared_error(y_train,y_pred_train)
valid_mse = mean_squared_error(y_test,y_pred_valid)
print('Train MSE : {0}'.format(train_mse))
print('Test  MSE : {0}'.format(valid_mse))

train_rmse = get_rmse(y_train,y_pred_train)
valid_rmse = get_rmse(y_test,y_pred_valid)
print('Train RMSE : {0}'.format(train_rmse))
print('Test  RMSE : {0}'.format(valid_rmse))

train_mae = mean_absolute_error(y_train,y_pred_train)
valid_mae = mean_absolute_error(y_test,y_pred_valid)
print('Train MAE : {0}'.format(train_mae))
print('Test  MAE : {0}'.format(valid_mae))

from scipy import stats
train_r2 = stats.pearsonr(np.array(y_train).flatten(), y_pred_train.flatten()).statistic**2
valid_r2 = stats.pearsonr(np.array(y_test).flatten(), y_pred_valid.flatten()).statistic**2
print('Train R2  : {0}'.format(train_r2))
print('Test  R2  : {0}'.format(valid_r2))



pd.DataFrame([train_r2,train_rmse,train_mae,valid_r2,valid_rmse,valid_mae]).T.to_clipboard(excel=True, sep=None, index=False, header=None)
#
#
# pd.concat([pd.DataFrame(y_train.reset_index(drop=True)),pd.DataFrame(y_pred_train)], axis=1, ignore_index=True).to_csv('result_DNN_trainset_wo_yellow_wo_meteo.csv', index=False)
# pd.concat([pd.DataFrame(y_test.reset_index(drop=True)),pd.DataFrame(y_pred_valid)], axis=1, ignore_index=True).to_csv('result_DNN_testset_wo_yellow_wo_meteo.csv', index=False)





import shap

explainer = shap.DeepExplainer(model, np.array(df_x))
explanation = explainer(np.array(df_x))
shap_values = explainer.shap_values(np.array(df_x))
shap_explanation = shap.Explanation(values=shap_values[:,:,0],
                                    data=df_x, feature_names=df_x.columns)
plt.figure()
shap.plots.beeswarm(shap_explanation, max_display=100, show=False)
plt.tight_layout()
plt.savefig('beeswarm.png')
plt.close()

plt.figure()
shap.plots.bar(shap_explanation, max_display=100, show=False)
plt.tight_layout()
plt.savefig('bar.png')
plt.close()

plt.figure()
plt.figure(figsize=(10,20))
shap.force_plot(shap_values, np.array(df_x), feature_names=df_x.columns, show=False)
plt.tight_layout()
plt.savefig('summary_plot.png')
plt.close()



