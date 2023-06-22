import pandas as pd 
from scipy.stats import bartlett
from scipy.stats import ttest_ind

co2 = pd.read_csv("CO2.csv")

chilled = co2[co2['Treatment']=="chilled"]
nonchilled = co2[co2['Treatment']=="nonchilled"]
## H0: Variances equal   H1:Variances Unequal
result = bartlett(chilled['uptake'], nonchilled['uptake'])
test_stat = result[0]
p_value = result[1]
print("Test Statistic =", test_stat)
print("P - Value =", p_value)

## Conclusion: We do not reject H0 at 5% l.o.s.
## Variances may be equal

result = ttest_ind(chilled['uptake'], nonchilled['uptake'],
          equal_var=True)
test_stat = result[0]
p_value = result[1]
print("Test Statistic =", test_stat)
print("P - Value =", p_value)

## Conclusion: We reject H0 at 5% l.o.s.
## Means may be different.

######### Lower tailed

result = ttest_ind(chilled['uptake'], nonchilled['uptake'],
          alternative="less", equal_var=True)
test_stat = result[0]
p_value = result[1]
print("Test Statistic =", test_stat)
print("P - Value =", p_value)

## Conclusion: We reject H0 at 5% l.o.s.
## Means of chilled may be less than mean of nonchilled

################# Puromycin ######################
puromycin = pd.read_csv("Puromycin.csv")
treated = puromycin[puromycin['state']=='treated']
untreated = puromycin[puromycin['state']=='untreated']

## H0: Variances equal   H1:Variances Unequal
result = bartlett(treated['rate'], untreated['rate'])
test_stat = result[0]
p_value = result[1]
print("Test Statistic =", test_stat)
print("P - Value =", p_value)

## Conclusion: We do not reject H0 at 5% l.o.s.
## Variances may be equal


result = ttest_ind(treated['rate'], untreated['rate'],
                   equal_var=True)
test_stat = result[0]
p_value = result[1]
print("Test Statistic =", test_stat)
print("P - Value =", p_value)

## Conclusion: We do not reject H0 at 5% l.o.s.
## Means may be equal.






