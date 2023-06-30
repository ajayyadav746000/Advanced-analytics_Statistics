import pandas as pd 

from scipy.stats import mannwhitneyu

co2 = pd.read_csv("CO2.csv")

chilled = co2[co2['Treatment']=="chilled"]
nonchilled = co2[co2['Treatment']=="nonchilled"]

result = mannwhitneyu(chilled['uptake'], nonchilled['uptake'])
test_stat = result[0]
p_value = result[1]
print("Test Statistic =", test_stat)
print("P - Value =", p_value)

## Conclusion: Means may be different


################# Puromycin ######################
puromycin = pd.read_csv("Puromycin.csv")
treated = puromycin[puromycin['state']=='treated']
untreated = puromycin[puromycin['state']=='untreated']

## H0: Variances equal   H1:Variances Unequal
result = mannwhitneyu(treated['rate'], untreated['rate'])
test_stat = result[0]
p_value = result[1]
print("Test Statistic =", test_stat)
print("P - Value =", p_value)

## Conclusion: We do not reject H0 at 5% l.o.s.
## Means may be equal