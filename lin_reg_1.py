import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression 
import numpy as np 

pizza = pd.read_csv("pizza.csv")

pizza['Promote'].corr(pizza['Sales'])
sns.scatterplot(data=pizza, x='Promote',
                y='Sales')
plt.show()

# X for independent variables; should be a 2D object
# y for dependent variable
X = pizza[['Promote']]
y = pizza['Sales']

lr = LinearRegression()
lr.fit(X, y)

print("Slope =", lr.coef_)
print("Intercept =", lr.intercept_)

tst_pizza = np.array([[16],[42],[29]])

# 23.50640302*tst_pizza + 5.4858653632529695
print( lr.predict(tst_pizza) )

sns.regplot(data=pizza, x='Promote', y='Sales', ci=0)
plt.show()

##################### Housing ####################

housing = pd.read_csv("Housing.csv")

housing['price'].corr(housing['lotsize'])

X = housing[['lotsize']]
y = housing['price']

lr = LinearRegression()
lr.fit(X, y)

print("Slope =", lr.coef_)
print("Intercept =", lr.intercept_)

sns.regplot(data=housing, x='lotsize',
            y='price', ci=0)
plt.show()

print(34136.19156491505 + 6.59876759*50)
print(34136.19156491505 + 6.59876759*51)

#############################
### X : Lot Size, Bedrooms

X = housing[['lotsize','bedrooms']]
y = housing['price']

lr = LinearRegression()
lr.fit(X, y)

print("Slopes =", lr.coef_)
print("Intercept =", lr.intercept_)

## 5000 sq ft, 3 bedrooms
print(5612.599731057446 + 6.05302208*5000 + 10567.3515*3)



