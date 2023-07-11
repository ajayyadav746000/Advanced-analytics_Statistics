import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np 
from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split 


sals = pd.read_csv("Exp_Salaries.csv")
dum_sals = pd.get_dummies(sals, drop_first=True)

train, test = train_test_split(dum_sals, test_size=0.3,
                               random_state=23)

X_train = train.drop('Salary', axis=1)
y_train = train['Salary']

lr = LinearRegression()
lr.fit(X_train, y_train)

X_test = test.drop('Salary', axis=1)
y_pred = lr.predict(X_test)

y_test = test['Salary']
print(r2_score(y_test, y_pred))


### different values of depth
depths = [2,3,4,5,6,7,8,9,10,11]
scores = []
for i in depths:
    dtr = DecisionTreeRegressor(random_state=23,
                                max_depth=i)
    dtr.fit(X_train, y_train)
    y_pred = dtr.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

max_score = np.max(scores)
i_max = np.argmax(scores)
best_depth = depths[i_max]
print("Best Depth =", best_depth)
print("Best Score =", max_score)

