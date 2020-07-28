import pandas as pd
import os

fpath = '/home/hdj/Desktop/实验记录/TimeCost/RDM/'

# df = pd.read_csv(fpath)
files = os.listdir(fpath)
t_df = pd.DataFrame()
seed = 0
for file in os.listdir(fpath):
    if not file.endswith('.csv'):
        continue

    path = fpath+file
    df = pd.read_csv(path)
    steps = df.head(20)['steps']
    t_df.insert(0,str(seed),steps)
    seed += 1

t_df.to_csv(fpath+'Total.csv')