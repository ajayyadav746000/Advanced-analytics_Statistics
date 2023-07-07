import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression 
import numpy as np 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sals = pd.read_csv("Exp_Salaries.csv")

X = sals.drop('Salary', axis=1)
y = sals['Salary']

X = pd.get_dummies(X, drop_first=True)

lr = LinearRegression()
lr.fit(X, y)

print("Slope =", lr.coef_)
print("Intercept =", lr.intercept_)


############################################

existing = np.array([56, 64, 94, 55, 22])
pred_1 = np.array([55.2, 45.33, 85.44, 54.22, 30])
print(mean_absolute_error(existing, pred_1))
print(mean_squared_error(existing, pred_1))
print(r2_score(existing, pred_1))

pred_2 = np.array([57, 58, 90, 53, 24])
print(mean_absolute_error(existing, pred_2))
print(mean_squared_error(existing, pred_2))
print(r2_score(existing, pred_2))

############### Housing ###################

housing = pd.read_csv("Housing.csv")
### X : Lot Size, Bedrooms

X = housing[['lotsize','bedrooms']]
y = housing['price']

lr = LinearRegression()
lr.fit(X, y)

print("Slopes =", lr.coef_)
print("Intercept =", lr.intercept_)

# tst_hous = np.array([[4000, 3],
#                      [5500, 2]])

# lr.predict(tst_hous)

y_pred_trn = lr.predict(X)
print(mean_absolute_error(y, y_pred_trn))
print(mean_squared_error(y, y_pred_trn))
print(r2_score(y, y_pred_trn))


### X : Lot Size, Bedrooms, Bathrooms, Storeys

X = housing[['lotsize','bedrooms','bathrms', 'stories']]
y = housing['price']

lr = LinearRegression()
lr.fit(X, y)

print("Slopes =", lr.coef_)
print("Intercept =", lr.intercept_)

y_pred_trn = lr.predict(X)
print(mean_absolute_error(y, y_pred_trn))
print(mean_squared_error(y, y_pred_trn))
print(r2_score(y, y_pred_trn))

################# Housing train test split #################

train, test = train_test_split(housing, test_size=0.3,
                               random_state=23)

X_train = train[['lotsize','bedrooms','bathrms', 'stories']]
y_train = train['price']

lr = LinearRegression()
lr.fit(X_train, y_train)

X_test = test[['lotsize','bedrooms','bathrms', 'stories']]
y_pred = lr.predict(X_test)

y_test = test['price']
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


## Entire

dum_hous = pd.get_dummies(housing, drop_first=True)
train, test = train_test_split(dum_hous, test_size=0.3,
                               random_state=23)

X_train = train.drop('price', axis=1)
y_train = train['price']

lr = LinearRegression()
lr.fit(X_train, y_train)

X_test = test.drop('price', axis=1)
y_pred = lr.predict(X_test)

y_test = test['price']
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

