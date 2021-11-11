from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# binary vector sampling func
# results = {
#     'ruletaker/runs/depth-5-base': ,
#     'ruletaker/runs/depth-5': ,
# }


res = results['ruletaker/runs/depth-5-base']

# x,y = zip(*res)
# plt.scatter(x,y)
# plt.savefig('basic.png')

df = pd.DataFrame(res)
df.columns = ['flip_rate', 'qlen']
summ = df.groupby('qlen').agg(['count', 'mean', 'std'])
