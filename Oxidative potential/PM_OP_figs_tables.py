
# 1:1 plots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
from sklearn.metrics import r2_score
from sklearn import linear_model
import sklearn

trainset = pd.read_csv('D:\\Github\\3.py_SCH\\result_DNN_trainset_w_yellow_w_meteo.csv')
testset = pd.read_csv('D:\\Github\\3.py_SCH\\result_DNN_testset_w_yellow_w_meteo.csv')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13
plt.rcParams.update({'figure.autolayout': True})

x = testset.iloc[:,0]
y = testset.iloc[:,1]

linreg = linear_model.LinearRegression()
# Fit the linear regression model
model = linreg.fit(np.array(x).reshape(-1,1), np.array(y).reshape(-1,1))

# Get the intercept and coefficients
intercept = model.intercept_
coef = model.coef_
result = [intercept, coef]
predicted_y = np.array(x).reshape(-1,1) * coef + intercept
r_squared = sklearn.metrics.r2_score(y, predicted_y)


plt.figure(figsize=(5, 5))
# plt.scatter(x, y, s=40, facecolors='none', edgecolors='k')
plt.plot(x, y, 'ko', markersize=8, mfc='none')
plt.plot(x, predicted_y, 'k-', 0.1, lw=1)
plt.xlim([0.0, 2.1])
plt.ylim([0.0, 2.1])
plt.plot([0, 2.1], [0, 2.1], 'k--', lw=2)
plt.text(1.15, 0.75,
         'y = %0.4fx + %0.4f \nR$^2$ = %0.2f' %(coef, intercept, r_squared))
plt.xlabel('Observed OPv')
plt.ylabel('Predicted OPv')
plt.show()

## Feature imfortances plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
fi = pd.read_csv('D:\\Github\\3.py_SCH\\Oxidation_potential_PM\\feature importances.csv')

col_names = ['Temperature', 'Relative humidity', 'Wind speed', 'NO$_2$', 'SO$_2$', 'O$_3$', 'CO', 'PM$_{2.5}$', 'PM$_{10}$',
       'OC', 'EC', 'WSTN', 'WSIN', 'WSON', 'WSOC', 'Cl$^-$', 'NO$_3$$^-$', 'SO$_4$$^{2-}$',
       'Na$^+$', 'NH$_4^+$', 'K$^+$', 'Ca$^{2+}$', 'Mg$^{2+}$', 'Li', 'Al', 'Ti', 'V', 'Cr',
       'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Cd', 'Pb']


fi_y1_m1_features = fi['Yellowdust'].iloc[1:]
fi_y0_m1_features = fi['Yellowdust.1'].iloc[1:]
fi_y1_m0_features = fi['Yellowdust.2'].iloc[1:]
fi_y0_m0_features = fi['Yellowdust.3'].iloc[1:]

fi_y1_m1_val1 = fi['Meteo'].str.split('±', expand=True).iloc[1:,0]
fi_y1_m1_val2 = fi['Meteo'].str.split('±', expand=True).iloc[1:,1]
fi_y0_m1_val1 = fi['Meteo.1'].str.split('±', expand=True).iloc[1:,0]
fi_y0_m1_val2 = fi['Meteo.1'].str.split('±', expand=True).iloc[1:,1]
fi_y1_m0_val1 = fi['Meteo.2'].str.split('±', expand=True).iloc[1:,0]
fi_y1_m0_val2 = fi['Meteo.2'].str.split('±', expand=True).iloc[1:,1]
fi_y0_m0_val1 = fi['Meteo.3'].str.split('±', expand=True).iloc[1:,0]
fi_y0_m0_val2 = fi['Meteo.3'].str.split('±', expand=True).iloc[1:,1]


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams.update({'figure.autolayout': True})
# Example data
index = np.arange(len(fi_y1_m1_features))*5
bar_width = 1
# Create the bar chart
plt.figure(figsize=(16, 6))
# plt.grid(axis='y', linestyle='--')
plt.grid(axis='x', linestyle='--')

plt.bar(index, fi_y1_m1_val1.astype('float64'), yerr=fi_y1_m1_val2.astype('float64'),
        capsize=2, label='w/ yellowdust and w/ meteorological data')
plt.bar(index+bar_width, fi_y1_m0_val1.astype('float64'), yerr=fi_y1_m0_val2.astype('float64'),
        capsize=2, label='w/ yellowdust and wo/ meteorological data')
plt.bar(index+bar_width*2, fi_y0_m1_val1.astype('float64'), yerr=fi_y0_m1_val2.astype('float64'),
        capsize=2, label='wo/ yellowdust and w/ meteorological data')
plt.bar(index+bar_width*3, fi_y0_m0_val1.astype('float64'), yerr=fi_y0_m0_val2.astype('float64'),
        capsize=2, label='wo/ yellowdust and wo/ meteorological data')
plt.bar(index+bar_width*4, np.zeros(36))

# Labeling the axes
# plt.xlabel('Chemical Species')
# plt.ylabel('Feature Importance')
plt.ylim([0,12])
plt.xlim([-1,181])
plt.xticks(index-1)
# Title for the plot
plt.legend()
# Display the plot
plt.tight_layout()  # Adjust layout to not cut off labels
plt.show()
