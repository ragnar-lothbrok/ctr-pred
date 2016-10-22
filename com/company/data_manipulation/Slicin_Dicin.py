import pandas as pd
import pandas as pd 
import matplotlib
import _tkinter
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py

matplotlib.use('TkAgg') 
plt.style.use('ggplot')

df = pd.read_csv("/home/raghunandangupta/gitPro/DAT210x/Module2/Datasets/census.data",sep=',', 
                  names = ['sNo','education', 'age', 'capital-gain', 'race', 'capital-loss', 'hours-per-week', 'sex', 'classification']);

# df = pd.read_csv("/home/raghunandangupta/gitPro/DAT210x/Module2/Datasets/servo.data",sep=',', 
#                   names = ['motor', 'screw', 'pgain', 'vgain', 'class']);

# print df.head(10)

print df.columns

df['sex'] = df['sex'].astype('category')
df['race'] = df['race'].astype('category')


cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

# df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
df[['capital-gain','capital-loss']] = df[['capital-gain','capital-loss']].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df = df.fillna(0)

df  = pd.get_dummies(df,columns=['education','classification'])

# print df.head(10)
# print df.dtypes

# plt.plot(df)
# plt.show()
# 
# 
# plt.plot(df.race)
# plt.show()


plt.scatter(df.race, df.sex)
plt.show()

# print  df[ df.vgain == 5].shape
# 
# 
# print  df[ (df.motor == 'E') &  (df.screw == 'E')].shape
# 
# print  df[ df.pgain == 4].describe()

# print df.loc[2:4,'col3']

# 
# print df
# 
# print df.recency
# 
# print df['recency']


# print df.loc[:, 'recency']

# print df.loc[:, ['recency']]
# 
# print df.iloc[:, 0]
# 
# print df.iloc[:, [0]]
# 
# print df.ix[:, 0]

# print df.ix[:, [0,1]]

# print df[['recency']]

# print df[0:2]

# print df.iloc[0:2, :]
# 
# print df.iloc[0:2, 0:6]

# print df.recency < 7

# print df[ (df.recency < 5) & (df.recency > 2) ]

# print df[df.recency <= 1]

# df[df.recency < 1] = -100
# print df
