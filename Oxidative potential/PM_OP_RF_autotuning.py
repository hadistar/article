import numpy as np
import pandas as pd
import warnings
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from hyperopt import tpe, hp, Trials
from hyperopt.fmin import fmin
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
def get_rmse(y_valid, y_predict):
    return np.sqrt(mean_squared_error(y_valid, y_predict))

# import dtale # EDA tool

df = pd.read_csv('D:\\OneDrive\\3.py_SCH\\data\\PM2.5_OP_2019-2021_2.csv')
#df = pd.read_csv('data/PM2.5_OP_2019-2021_2.csv')

#df_y = df[['OPv (DTTv)', 'OPm (DTTm)']]
df_y = df[['OPv (DTTv)']]

df_x = df[['온도', '습도', '풍속', 'O3', 'NO2', 'CO', 'PM2.5', 'PM10',
       'OC', 'EC', 'TC', 'WSTN', 'WSIN', 'WSON', 'WSOC', 'Cl', 'NO3', 'SO4',
       'Na', 'NH4', 'K', 'Ca', 'Mg', 'acidity', 'Li', 'Al', 'Ti', 'V', 'Cr',
       'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Cd', 'Pb', 'SO2']]

df_x.isna().sum().sum()

imputer = KNNImputer(n_neighbors=3) #KNN
df_x.iloc[:,:] = imputer.fit_transform(df_x)

df_x.isna().sum().sum()

# r,_=pearsonr(df_y.iloc[:,0],df_y.iloc[:,1])
# print(r**2)

df = pd.concat([df_x,df_y], axis=1)

scalingfactor = {}
data_scaled = df.copy()

for c in df.columns[:]:
    denominator = df[c].max()-df[c].min()
    scalingfactor[c] = [denominator, df[c].min(), df[c].max()]
    data_scaled[c] = (df[c] - df[c].min())/denominator

data_wodate_scaled = data_scaled.iloc[:, :]

# target = ['OPv (DTTv)', 'OPm (DTTm)']
target = ['OPv (DTTv)']

df_y = df[['OPv (DTTv)']]

df_x = df[['온도', '습도', '풍속', 'O3', 'NO2', 'CO', 'PM2.5', 'PM10',
       'OC', 'EC', 'TC', 'WSTN', 'WSIN', 'WSON', 'WSOC', 'Cl', 'NO3', 'SO4',
       'Na', 'NH4', 'K', 'Ca', 'Mg', 'acidity', 'Li', 'Al', 'Ti', 'V', 'Cr',
       'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Cd', 'Pb', 'SO2']]

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=7)

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


##하이퍼 파라미터 최적화
def objective(params):
    est = int(params['n_estimators'])
    md = int(params['max_depth'])
    msl = int(params['min_samples_leaf'])
    mss = int(params['min_samples_split'])
    mf = params['max_features']
    model = RandomForestRegressor(n_estimators=est, max_depth=md, min_samples_leaf=msl,
                                  min_samples_split=mss, max_features=mf, n_jobs=-1)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = mean_squared_error(y_test, pred)
    return score


def optimize(trial):
    params = {'n_estimators': hp.uniform('n_estimators', 1, 300),
              'max_depth': hp.uniform('max_depth', 1, 30),
              'min_samples_leaf': hp.uniform('min_samples_leaf', 1, 10),
              'min_samples_split': hp.uniform('min_samples_split', 2, 10),
              'max_features': hp.choice('max_features', [1,30])
              }
    best = fmin(fn=objective, space=params, algo=tpe.suggest, trials=trial, max_evals=200)
    return best


trial = Trials()
best = optimize(trial)


best_n_estimators = round(best['n_estimators'])
best_max_depth = round(best['max_depth'])
best_min_samples_leaf = round(best['min_samples_leaf'])
best_min_samples_split = round(best['min_samples_split'])
best_max_features = best['max_features']

best_model = RandomForestRegressor(n_estimators=best_n_estimators,
                                   max_depth=best_max_depth,
                                   min_samples_leaf=best_min_samples_leaf,
                                   min_samples_split=best_min_samples_split,
                                   n_jobs=-1)

best_model.fit(x_train, y_train)


y_pred_train = best_model.predict(x_train.values)
y_pred_valid = best_model.predict(x_test.values)
print("Y Pred Train Shape : ", y_pred_train.shape)
print("Y Pred Test  Shape : ", y_pred_valid.shape)

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

train_r2 = r2_score(y_train, y_pred_train)
valid_r2 = r2_score(y_test, y_pred_valid)
print('Train R2  : {0}'.format(train_r2))
print('Test  R2  : {0}'.format(valid_r2))

y_predicted = best_model.predict(x_test)
evaluation = best_model.score(x_test, y_test)

f = open('result_.txt', 'w')
f.write(f"""
    The hyperparameter search is complete. 
    The best hyperparameters
     - n_estimators: {best_n_estimators}
     - max_depth: {best_max_depth}
     - min_samples_leaf: {best_min_samples_leaf}
     - min_samples_split: {best_min_samples_split}
     - max_features: {best_max_features}
    R2 = {evaluation}.
    Y Pred Train Shape : {y_pred_train.shape}
    Y Pred Test  Shape : {y_pred_valid.shape}
    Train MSE : {train_mse}
    Test  MSE : {valid_mse}
    Train RMSE : {train_rmse}
    Test  RMSE : {valid_rmse}
    Train MAE : {train_mae}
    Test  MAE : {valid_mae}
    Train R2  : {train_r2}
    Test  R2  : {valid_r2}
    """)
f.close()

# rescaling
# x = x' * (max-min) + min
# saving scaling factor in [max-min, min, max]

y_predicted_total = best_model.predict(np.array(data_wodate_scaled.drop(columns=target)))
y_predicted_total = pd.DataFrame(y_predicted_total, columns=target)

for c in y_predicted_total:
    y_predicted_total[c] = y_predicted_total[c] * scalingfactor[c][0] + scalingfactor[c][1]

y_predicted_total.to_csv('result_test_.csv', index=False)



# Feature importance

feature_importance = best_model.feature_importances_

fi=pd.concat([pd.DataFrame(df_x.columns), pd.DataFrame(feature_importance)], axis=1)

