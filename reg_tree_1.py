import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
import numpy as np 
from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split 

housing = pd.read_csv("Housing.csv")

dum_hous = pd.get_dummies(housing, drop_first=True)
train, test = train_test_split(dum_hous, test_size=0.3,
                               random_state=23)

X_train = train.drop('price', axis=1)
y_train = train['price']

X_test = test.drop('price', axis=1)
y_test = test['price']

dtr = DecisionTreeRegressor(random_state=23,
                            max_depth=2)
dtr.fit(X_train, y_train)
### Drawing a tree

plt.figure(figsize=(30,10))
tree.plot_tree(dtr,feature_names=X_train.columns,
               filled=True,fontsize=16) 
plt.show()

############# Getting the parts of the tree
# y_train.mean()

# left_ds = train[train['lotsize']<=5954]
# right_ds = train[train['lotsize']>5954]

# left_ds['price'].mean()
# right_ds['price'].mean()

# np.mean((y_train - y_train.mean())**2)

############ Predicting on the test set
y_pred = dtr.predict(X_test)
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
