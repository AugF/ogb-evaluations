import os.path as osp
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
        elif name == "coauthor-physics":
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


if __name__ == "__main__":
    # datasets = ["cora", "pubmed", "citeseer", "amazon-computers", "amazon-photo", "coauthor-physics", "flickr"]
    # for data in datasets:
    #     get_dataset(data)
    import sys
    
    data = sys.argv[1]
    get_dataset(data)