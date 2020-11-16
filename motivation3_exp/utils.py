import os.path as osp
import numpy as np
import json
import torch
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets.amazon import Amazon
from torch_geometric.datasets.coauthor import Coauthor
from torch_geometric.datasets import Flickr

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


if __name__ == "__main__":
    test()