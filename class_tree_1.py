import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split 

mowers = pd.read_csv("RidingMowers.csv")
##################
sns.scatterplot(data=mowers,
                x='Income',y='Lot_Size',
                hue='Response')
plt.show()
##############

train, test = train_test_split(mowers, test_size=0.3,
                               random_state=23,
                               stratify=mowers['Response'])

X_train = train.drop('Response', axis=1)
y_train = train['Response']

X_test = test.drop('Response', axis=1)
y_test = test['Response']

dtc = DecisionTreeClassifier(random_state=23,
                             max_depth=3)
dtc.fit(X_train, y_train)

### Drawing a tree

plt.figure(figsize=(25,10))
tree.plot_tree(dtc,feature_names=X_train.columns,
               filled=True,fontsize=16,
               class_names=['Bought', 'Not Bought']) 
plt.show()

#### Predicting on test data
y_pred = dtc.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

### Heat map of Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=dtc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=dtc.classes_)
disp.plot()

plt.show(