import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency

################ Exploratory Data Analysis ##################

big_mart = pd.read_csv(r"C:\Training\AV\Big Mart III\processed_train.csv")

big_mart['i_weight'].corr(big_mart['Item_MRP'])
sns.scatterplot(data=big_mart, x='i_weight',y='Item_MRP')
plt.show()

big_mart['Item_Outlet_Sales'].corr(big_mart['Item_MRP'])
sns.scatterplot(data=big_mart, x='Item_MRP',y='Item_Outlet_Sales')
plt.show()

ols_1 = ols('Item_Outlet_Sales ~ Item_Type', data=big_mart).fit()
table = anova_lm(ols_1, typ=2)
print(table)

# Boxplot
sns.boxplot(x="Item_Type",y='Item_Outlet_Sales', data=big_mart)
plt.xticks(rotation=60)
plt.show()

# Bar Plot
cts = big_mart.groupby('Item_Type')['Item_Outlet_Sales'].mean()
plt.bar(cts.index, cts)
plt.xticks(rotation=60)
plt.show()


ctab = pd.crosstab(index=big_mart['Item_Type'], 
            columns=big_mart['Item_Fat_Content'])

test_statistic, p_value, df, expected_frequencies = chi2_contingency(ctab)
print("P-Value =", p_value)


df_bar = pd.melt(ctab.reset_index(),id_vars="Item_Type")
sns.barplot(y="Item_Type",
           x="value",
           hue="Item_Fat_Content",
           data=df_bar)
plt.title("Grouped Bar Chart")
plt.show()



