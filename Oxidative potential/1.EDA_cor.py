import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import seaborn as sns

## 0. Data loading

df = pd.read_csv('D:\\OneDrive\\3.py_SCH_data\\data_OP_ML\\PM2.5_OP_2019-2021_2.csv')

## 1. Summary statistics table

df_stats = df[['OPv (DTTv)', '온도', '습도', '풍속', 'NO2', 'SO2', 'O3', 'CO', 'PM2.5', 'PM10',
       'OC', 'EC', 'WSTN', 'WSIN', 'WSON', 'WSOC', 'Cl', 'NO3', 'SO4',
       'Na', 'NH4', 'K', 'Ca', 'Mg', 'Li', 'Al', 'Ti', 'V', 'Cr',
       'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Cd', 'Pb']]
df_stats.columns = ['OPv (DTTv)', 'Temperature', 'Relative humidity', 'Wind speed', 'NO$_2$', 'SO$_2$', 'O$_3$', 'CO', 'PM$_{2.5}$', 'PM$_{10}$',
       'OC', 'EC', 'WSTN', 'WSIN', 'WSON', 'WSOC', 'Cl$^-$', 'NO$_3$$^-$', 'SO$_4$$^{2-}$',
       'Na$^+$', 'NH$_4^+$', 'K$^+$', 'Ca$^{2+}$', 'Mg$^{2+}$', 'Li', 'Al', 'Ti', 'V', 'Cr',
       'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Cd', 'Pb']

df_stats = df_stats.describe().T.to_clipboard()


## 2. Pearson coefficient heatmap

# for Pearson coefficient heatmap figure
df_pearson = df[['OPv (DTTv)', '온도', '습도', '풍속', 'NO2', 'SO2', 'O3', 'CO', 'PM2.5', 'PM10',
       'OC', 'EC', 'WSTN', 'WSIN', 'WSON', 'WSOC', 'Cl', 'NO3', 'SO4',
       'Na', 'NH4', 'K', 'Ca', 'Mg', 'Li', 'Al', 'Ti', 'V', 'Cr',
       'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Cd', 'Pb']]
df_pearson.columns = ['OPv (DTTv)', 'Temperature', 'Relative humidity', 'Wind speed', 'NO$_2$', 'SO$_2$', 'O$_3$', 'CO', 'PM$_{2.5}$', 'PM$_{10}$',
       'OC', 'EC', 'WSTN', 'WSIN', 'WSON', 'WSOC', 'Cl$^-$', 'NO$_3$$^-$', 'SO$_4$$^{2-}$',
       'Na$^+$', 'NH$_4^+$', 'K$^+$', 'Ca$^{2+}$', 'Mg$^{2+}$', 'Li', 'Al', 'Ti', 'V', 'Cr',
       'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Cd', 'Pb']

corr = df_pearson.corr()

plt.figure(figsize=(16,12))
sns.set_theme(style="white")
ax = sns.heatmap(corr, cmap='coolwarm',vmin=-1.0, vmax=1.0, annot_kws={"size": 30})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel('Variables', fontsize=20)
plt.ylabel('Variables', fontsize=20)
plt.tight_layout()
plt.show()


