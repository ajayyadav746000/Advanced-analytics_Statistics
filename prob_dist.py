from scipy.stats import binom, poisson, norm
import numpy as np

binom.pmf(5,40,0.25)

# probs = []
# for i in range(0,41):
#     probs.append(binom.pmf(i,40,0.25))

probs = binom.pmf(np.arange(0,41),40,0.25)
print(np.sum(probs))

# probs = binom.pmf(np.arange(0,11),40,0.25)
# print(np.sum(probs))

binom.cdf(10,40,0.25)

binom.sf(19, 40, 0.25)

binom.stats(40, 0.25)
#####################1.####################

binom.pmf(5,20,0.15)

binom.sf(12,20,0.15)

# probs = binom.pmf(np.arange(13,21),20,0.15)
# print(np.sum(probs))

binom.cdf(10,20,0.15)

######### Poisson ##############
poisson.pmf(5, 12)

poisson.cdf(12,12)

poisson.sf(14, 12)

probs = poisson.pmf(np.arange(10,16), 12)
np.sum(probs)

poisson.cdf(15, 12) - poisson.cdf(9, 12)

####### 1. #######
poisson.sf(70, 56)
poisson.cdf(19, 56)

############## Normal ##############
norm.cdf(58, 64, 4)
norm.sf(200, 180, 30)

norm.ppf(0.95, 100, 15)

##########
norm.ppf(0.9, 1678, 500)

norm.cdf(1900, 1678, 500) - norm.cdf(1000, 1678, 500)

norm.ppf(0.98, 313, 57)

norm.ppf(0.9, 93, 22)

# 1st Q
norm.ppf(0.25, 93, 22)
# 2nd Q
norm.ppf(0.5, 93, 22)
# 3rd Q
norm.ppf(0.75, 93, 22)

import pandas as pd
import numpy as np
cars93 = pd.read_csv("Cars93.csv")
np.mean(cars93['Price'])
### Equivalent to ppf
np.quantile(cars93['Price'], 0.9)

np.quantile(cars93['Price'], 0.25)
cars93['Type'].value_counts()
cars93['AirBags'].value_counts()
pd.crosstab(index=cars93['AirBags'], 
            columns=cars93['Type'])
#######################################

pa = norm.sf(450,313,57)
pb = norm.sf(150,93,22)

print(pa+pb-(pa*pb))












