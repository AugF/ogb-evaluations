import os, sys, re
import pandas as pd
from utils import run_batches, percents

base_path = "/home/wangzhaokang/wangyunpan/gnns-project/ogb-evaluations/motivation3_exp"
os.chdir(base_path)

if not os.path.exists(base_path + "/log"):
    os.makedirs(base_path + "/log")
    
if not os.path.exists(base_path + "/res"):
    os.makedirs(base_path + "/res")

df_data = {}
for data in run_batches.keys():
# for data in ['coauthor-cs']:
    print(f"begin: {data}")
    df_data[data] = []
    for bs in run_batches[data]:
        # print(f"begin: {bs}")
        # cmd = f"python -u graphsaint.py --batch_size {bs} --dataset {data} --device 0 --epochs 500 1>log/graphsaint_{data}_{bs}.log 2>&1"
        # os.system(cmd)
        with open(f"log/graphsaint_{data}_{bs}.log") as f:
            for line in f:
                match_line = re.match("Final Test: (.*)", line)
                if match_line:
                    df_data[data].append(float(match_line.group(1)))
                    break
    if df_data[data] == []:
        del df_data[data]

print(df_data)
pd.DataFrame(df_data, index=[str(i * 100) + '%' for i in percents]).to_csv("res/graphsaint.csv")