import os
import random

os.chdir("/home/wangzhaokang/wangyunpan/gnns-project/ogb_evaluations/exp")
for data in ['mag', 'products']:
    for sampler in ['neighborsampling', 'cluster', 'graphsaint']:
        for alg in ['gcn', 'sage', 'gat', 'rgcn']:
            file_path = f"{data}/{sampler}_{alg}.py"
            if not os.path.exists(file_path):
                continue
            print(file_path)
            cmd = f"python -u {file_path} --device {random.randint(0,1)} 1>log/{data}_{sampler}_{alg}.out 2>&1"
            os.system(cmd)

