import os

from utils import batch_sizes

os.chdir("/home/wangzhaokang/wangyunpan/gnns-project/ogb_evaluations/exp/products")

file_name = "products_neighborsampling_sage"

for bs in batch_sizes[file_name]:
    os.system(f"python neighborsampling_sage.py --batch_size {bs}")