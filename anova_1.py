import pandas as pd 
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

agr = pd.read_csv("Yield.csv")
agr.groupby('Treatments').mean()

agrYield = ols('Yield ~ Treatments', data=agr).fit()
table = anova_lm(agrYield, typ=2)
print(table)

### Post hoc test : Tukey's HSD Test
compare = pairwise_tukeyhsd(agr['Yield'], 
                            agr['Treatments'], alpha=0.05)
pd.DataFrame(compare._results_table.data)

############### Plant Growth #############################
plant = pd.read_csv("PlantGrowth.csv")

ols_plant = ols('weight ~ group', data=plant).fit()
table = anova_lm(ols_plant, typ=2)
print(table)

### Tukey's Test
compare = pairwise_tukeyhsd(plant['weight'], 
                            plant['group'], alpha=0.05)
pd.DataFrame(compare._results_table.data)




