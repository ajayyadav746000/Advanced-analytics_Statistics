import pandas as pd 
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

cars93 = pd.read_csv("Cars93.csv")

pd.crosstab(index=cars93['Type'], 
            columns=cars93['AirBags'],
            margins=True)

ctab = pd.crosstab(index=cars93['Type'], 
            columns=cars93['AirBags'])

test_statistic, p_value, df, expected_frequencies = chi2_contingency(ctab)
print("P-Value =", p_value)
n

df_bar = pd.melt(ctab.reset_index(),id_vars="Type")
sns.barplot(x="AirBags",
           y="value",
           hue="Type",
           data=df_bar)
plt.title("Grouped Bar Chart")
plt.show()