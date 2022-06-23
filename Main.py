import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('archive/Housing.csv')
df_copy = df.copy(deep=True)
df.head()


df.info()

df.describe()

df.furnishingstatus.unique()
cat_columns = ['mainroad',
               'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
               'prefarea']


def binary_mapping(x):
    return x.map({'yes': 1, "no": 0})


df[cat_columns] = df[cat_columns].apply(binary_mapping)

df.head()

ohe = OneHotEncoder(sparse=False, handle_unknown='error', drop='first')
ohe_df = pd.DataFrame(ohe.fit_transform(df[['furnishingstatus']]))

ohe_df.columns = ohe.get_feature_names(['status'])

ohe_df.head()
df = pd.concat([df, ohe_df], axis=1)
df.drop(['furnishingstatus'], axis=1, inplace=True)
df.head()

sns.pairplot(df)
plt.show()

sns.countplot(y='price', data=df_copy)
plt.show()

sns.lineplot(x='area', y='price', data=df_copy)
plt.show()

sns.relplot(x='area', y='price', hue='mainroad', size='parking', data=df_copy)

sns.relplot(x='area', y='price', hue='airconditioning', size='parking', data=df_copy)

sns.relplot(x='area', y='price', hue='guestroom', size='parking', data=df_copy)

sns.relplot(x='area', y='price', hue='basement', size='parking', data=df_copy)

sns.relplot(x='area', y='price', hue='hotwaterheating', size='parking', data=df_copy)

sns.catplot(x='price', y='furnishingstatus', data=df_copy)

sns.catplot(x='furnishingstatus', kind='count', data=df_copy, orient='h')

df_new = df.copy(deep=True)
num_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

df_new[num_columns] = MinMaxScaler().fit_transform(df_new[num_columns])

df_new.head()
y = df_new.pop('price')
x = df_new
np.random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
model.coef_
model.intercept


y_pred = model.predict(x_test)

plt.scatter(y_test, y_pred, c='maroon')
x = y
plt.plot(x, y, c='k')
plt.title('y_test versus y_pred')
plt.xlabel('Test value--->')
plt.ylabel('Predicted value--->')
plt.show()

per_error = 100 * (y_pred - y_test) / y_test

df_prd_tst = pd.DataFrame({'Predicted Price': y_pred.astype('int64'), 'Actual Price': y_test, '% Error': per_error})
df_prd_tst.to_csv('predictions.csv')


abs(per_error).max()

abs(per_error).min()

abs(per_error).mean()

mean_squared_error(y_test, y_pred, squared=False)

r2_score(y_test, y_pred)
