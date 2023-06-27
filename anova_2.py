import pandas as pd 
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

cars93 = pd.read_csv("Cars93.csv")

### Price and AirBags
ols_air = ols('Price ~ AirBags', data=cars93).fit()
table = anova_lm(ols_air, typ=2)
print(table)


### Post hoc test : Tukey's HSD Test
compare = pairwise_tukeyhsd(cars93['Price'], 
                            cars93['AirBags'], alpha=0.05)
pd.DataFrame(compare._results_table.data)

# Boxplot
sns.boxplot(x="AirBags",y='Price', data=cars93)
plt.show()

# Bar Plot
cts = cars93.groupby('AirBags')['Price'].mean()
plt.bar(cts.index, cts)
plt.show()

### Price and Origin
ols_org = ols('Price ~ Origin', data=cars93).fit()
table = anova_lm(ols_org, typ=2)
print(table)


### Price and Type
ols_type = ols('Price ~ Type', data=cars93).fit()
table = anova_lm(ols_type, typ=2)
print(table)


### Post hoc test : Tukey's HSD Test
compare = pairwise_tukeyhsd(cars93['Price'], 
                            cars93['Type'], alpha=0.05)
pd.DataFrame(compare._results_table.data)
