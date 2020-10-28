import time
import os
import sys

batch_sizes = {'mag_cluster_rgcn': [50, 150, 300, 500, 1250],
               'mag_graphsaint_rgcn': [1939, 9698, 19397, 58192, 116384],
               'mag_neighborsampling_rgcn': [969, 1939, 9698, 19397, 58192],
               'products_cluster_gcn': [150, 450, 900, 1500, 3750],
               'products_cluster_sage': [150, 450, 900, 1500, 3750],
               'products_graphsaint_sage': [244, 1224, 2449, 12245, 24490]}


os.chdir("/home/wangzhaokang/wangyunpan/gnns-project/ogb_evaluations/exp")

if len(sys.argv) < 3:
    print("python run_relative_percent.py data device-id")
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
        for bs in batch_sizes[f'{data}_{sampler}_{alg}']:
            print("batch_size", bs)
            st = time.time()
            cmd = f"python -u {file_path} --epochs 75 --runs 2 --device {device_id} 1>log/{data}_{sampler}_{alg}_{bs}.out 2>&1"
            os.system(cmd)
            ed = time.time()
            print(f"{time.strftime('%Y.%m.%d',time.localtime(ed))}, use_time: {ed - st}")