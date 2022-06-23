import numpy as np  # Linear algera Library
import pandas as pd
import matplotlib.pyplot as plt  # to plot graphs
import seaborn as sns  # to plot graphs
from sklearn.linear_model import LinearRegression  # for linear regression model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import warnings

sns.set()  # setting seaborn as default

warnings.filterwarnings('ignore')

data = pd.read_csv('archive/Housing.csv')  # reads the input data
data.head()

data.info()
data.describe(include='all')
data.isnull().sum()
categorical = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']


# write a function to change yes to 1 and no to 0
def binary_map(x):
    return x.map({'yes': 1, "no": 0})


# now replace yes and no with 1 and 0 in our dataset
data[categorical] = data[categorical].apply(binary_map)
data.head()

table = pd.get_dummies(data['furnishingstatus'])  # add the column into table variable
table.head()
table = pd.get_dummies(data['furnishingstatus'],
                       drop_first=True)  # recreate table but now drop the first column(furnished)
table.head()
data = pd.concat([data, table], axis=1)  # attach the other two columns to our data set
data.head()
data.drop(['furnishingstatus'], axis=1, inplace=True)  # drop the old column from the dataset
data.head()
sns.pairplot(data)
plt.show()
data.columns

np.random.seed(0)  # so data can have same values
df_train, df_test = train_test_split(data, train_size=0.7, test_size=0.3, random_state=100)
df_train.head()


scaler = MinMaxScaler()
var_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
df_train[var_to_scale] = scaler.fit_transform(df_train[var_to_scale])

df_train.head()
df_train.describe()
y_train = df_train.pop('price')
x_train = df_train
y_train.head()
lm = LinearRegression()
lm.fit(x_train, y_train)

lm.coef_

lm.score(x_train, y_train)
var_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
df_test[var_to_scale] = scaler.fit_transform(df_test[var_to_scale])

y_test = df_test.pop('price')
x_test = df_test
predictions = lm.predict(x_test)

r2_score(y_test, predictions)
y_test.shape
y_test_matrix = y_test.values.reshape(-1, 1)
dframe = pd.DataFrame({'actual': y_test_matrix.flatten(), 'Predicted': predictions.flatten()})
dframe.head(15)
fig = plt.figure()
plt.scatter(y_test, predictions)
plt.title('Actual versus Prediction ')
plt.xlabel('Actual', fontsize=20)
plt.ylabel('Predicted', fontsize=20)
# trying the same with a reg plot(optonal)
sns.regplot(y_test, predictions)
plt.title('Actual versus Prediction ')
plt.xlabel('Actual', fontsize=20)
plt.ylabel('Predicted', fontsize=20)
