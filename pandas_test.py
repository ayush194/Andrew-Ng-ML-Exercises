# Render our plots inline
# %matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
plt.rcParams['figure.figsize'] = (15, 5)

broken_df = pd.read_csv('./data/ex1data1.txt.')
# Look at the first 3 rows
print(broken_df[:3])
