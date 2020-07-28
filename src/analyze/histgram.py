import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mean = pd.read_csv('/home/hdj/Desktop/实验记录/DistanceCost/mean.csv')
conf = pd.read_csv('/home/hdj/Desktop/实验记录/DistanceCost/confidence.csv')

dme_drl = mean.loc[0,['Map %d'%i for i in range(10)]].to_numpy()
maddpg = mean.loc[1,['Map %d'%i for i in range(10)]].to_numpy()
rf = mean.loc[2,['Map %d'%i for i in range(10)]].to_numpy()
nf = mean.loc[3,['Map %d'%i for i in range(10)]].to_numpy()

dme_drl_conf = conf.loc[0,['Map %d'%i for i in range(10)]].to_numpy()
maddpg_conf = conf.loc[1,['Map %d'%i for i in range(10)]].to_numpy()
rf_conf = conf.loc[2,['Map %d'%i for i in range(10)]].to_numpy()
nf_conf = conf.loc[3,['Map %d'%i for i in range(10)]].to_numpy()

total_width, n = 0.8, 4
width = total_width/n
x = np.arange(0,10)

data_mean = [dme_drl,maddpg,rf,nf]
data_conf = [dme_drl_conf,maddpg_conf,rf_conf,nf_conf]

methods = ['DME-DRL','MADDPG','RF','NF']
colors = ['#32D3EB','#FEB64D','#FF7C7C','#9287E7']
for i in range(len(data_mean)):
    if i%2 == 0:
        plt.bar(x+i*width,data_mean[i],width,label=methods[i],tick_label=x,yerr=data_conf[i],fc=colors[i])
    else:
        plt.bar(x+i*width,data_mean[i],width,label=methods[i],yerr=data_conf[i],fc=colors[i])
plt.xlabel('Map')
plt.ylabel('Distance Traveled')
plt.legend(loc='upper left')
plt.show()