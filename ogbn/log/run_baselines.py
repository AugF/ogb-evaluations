import os
import sys
import time

os.chdir("/home/wangzhaokang/wangyunpan/gnns-project/ogb_evaluations/ogbn")

if len(sys.argv) < 3:
    print("python run_baselines.py data device-id")
    sys.exit(0)
    
data = sys.argv[1]
device_id = sys.argv[2]

# for data in ['mag', 'products']:
for sampler in ['neighborsampling', 'cluster', 'graphsaint']:
    for alg in ['gcn', 'sage', 'gat', 'rgcn']:
        file_path = f"{data}/{sampler}_{alg}.py"
        if not os.path.exists(file_path):
            continue
        print(file_path)
        st = time.time()
        cmd = f"python -u {file_path} --device {device_id} 1>log/{data}_{sampler}_{alg}.out 2>&1"
        os.system(cmd)
        ed = time.time()
        print(f"{time.strftime('%Y.%m.%d',time.localtime(ed))}, use_time: {ed - st}")
        

