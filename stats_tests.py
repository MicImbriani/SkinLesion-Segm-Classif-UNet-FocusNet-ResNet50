import numpy as np
import pylab
import scipy.stats as stats


# QQ PLOTS
results = ""
stats.propbplot(results, dist="norm", plot=pylab)
pylab.show()

# BOX AND WHISKERS
import seaborn as sns
results = ""
ax = sns.bocplot(x=results)