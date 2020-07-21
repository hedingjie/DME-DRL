import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

data = pd.read_csv('/home/hdj/Git/DME-DRL/src/results/comparision (copy).csv')
sn.barplot(x='Map ID', y='Distance Traveled', hue='Method', data=data)
plt.show()
