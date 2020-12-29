import os.path as osp
import numpy as np
import json
import torch
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets.amazon import Amazon
from torch_geometric.datasets.coauthor import Coauthor
from torch_geometric.datasets import Flickr

percents = [0.01, 0.05, 0.1, 0.25, 0.50]
nodes = [2708, 3327, 19717, 7650, 13752, 34493, 18333, 89250]
run_batches = {
    'cora': [27, 135, 270, 677, 1354], 
    'citeseer': [33, 166, 332, 831, 1663], 
    'pubmed': [197, 985, 1971, 4929, 9858], 
    'amazon-photo': [76, 382, 765, 1912, 3825], 
    'amazon-computers': [137, 687, 1375, 3438, 6876], 
    'coauthor-physics': [344, 1724, 3449, 8623, 17246], 
    'coauthor-cs': [183, 916, 1833, 4583, 9166], 
    'flickr': [892, 4462, 8925, 22312, 44625]}


def get_dataset(name, normalize_features=False, transform=None): #
    if name in ["cora", "pubmed", "citeseer"]: 
        path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets')
        dataset = Planetoid(path, name, split='full')
    else:
        path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets', name)
        if name in ["amazon-computers", "amazon-photo"]:
            dataset = Amazon(path, name[7:])
        elif name in ["coauthor-cs", "coauthor-physics"]: 
            dataset = Coauthor(path, name[9:])
        elif name == "flickr":  # flickr, reddit?, yelp?
            dataset = Flickr(path)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def gen_roles(raw_dir, nodes, tr=0.70, va=0.15, seed=1):
    np.random.seed(seed)
    idx = np.arange(nodes)
    np.random.shuffle(idx)
    tr, va = int(nodes * tr), int(nodes * (tr + va))
    role = {'tr': idx[: tr].tolist(),
            'va': idx[tr: va].tolist(),
            'te': idx[va:].tolist()
            }

    with open(raw_dir + "/role.json", "w") as f:
        json.dump(role, f)


def get_split_by_file(file_path, nodes): # 通过读取roles.json文件来获取train, val, test mask
    with open(file_path) as f:
        role = json.load(f)

    train_mask = torch.zeros(nodes, dtype=torch.bool)
    train_mask[torch.tensor(role['tr'])] = True

    val_mask = torch.zeros(nodes, dtype=torch.bool)
    val_mask[torch.tensor(role['va'])] = True

    test_mask = torch.zeros(nodes, dtype=torch.bool)
    test_mask[torch.tensor(role['te'])] = True
    return train_mask, val_mask, test_mask


def test():
    datasets = ['cora', 'citeseer', 'pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'coauthor-cs', 'flickr']
    for data in datasets:
        dataset = get_dataset(data, normalize_features=False)[0]
        print(data, dataset)


gen_roles(raw_dir="/home/wangzhaokang/wangyunpan/gnns-project/datasets/coauthor-cs/raw/", nodes=18333, tr=0.70, va=0.15, seed=1)