import pandas as pd
from scipy.stats import ttest_rel

rhuem = pd.read_csv("Rheumatic.csv")

result = ttest_rel(rhuem['Before'], rhuem['After'],
          alternative='less')

test_stat = result[0]
p_value = result[1]
print("Test Statistic =", test_stat)
print("P - Value =", p_value)

## Conclusion: We reject H0 at 5% level of significance
## There may be an improvement in breathing capacity