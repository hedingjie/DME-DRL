import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

data = pd.read_csv('/home/hdj/Desktop/实验记录/DistanceCost/comparision (copy).csv')
data.sort_values(by='Distance Traveled')
sn.barplot(x=data['Map ID'], y=data['Distance Traveled'], hue=data['Method'], data=data, capsize=.2)
plt.show()
