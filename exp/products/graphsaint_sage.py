'''
https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/graph_saint.py master: cb1f235
report acc: 0.7908 ± 0.0024
rank: 10
date: 2020-10-27 
'''
import time
import sys
import argparse

import torch
from tqdm import tqdm
import torch.nn.functional as F

from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import subgraph

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return torch.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        # pbar = tqdm(total=x_all.size(0) * len(self.convs))
        # pbar.set_description('Evaluating')
        total_batches = len(subgraph_loader)
        
        sampling_time, to_time, train_time = 0.0, 0.0, 0.0
        
        for i, conv in enumerate(self.convs):
            xs = []
            
            loader_iter = iter(subgraph_loader)
            while True:
                try:
                    et0 = time.time()
                    batch_size, n_id, adj = next(loader_iter)
                    et1 = time.time()
                    edge_index, _, size = adj.to(device)
                    x = x_all[n_id].to(device)
                    et2 = time.time()
                    x_target = x[:size[1]]
                    x = conv((x, x_target), edge_index)
                    if i != len(self.convs) - 1:
                        x = F.relu(x)
                    xs.append(x.cpu())
                    et3 = time.time()
                    
                    sampling_time += et1 - et0
                    to_time += et2 - et1
                    train_time += et3 - et2
                except StopIteration:
                    break
                
            x_all = torch.cat(xs, dim=0)

        sampling_time /= total_batches
        to_time /= total_batches
        train_time /= total_batches
        print(f"Evaluation: sampling time: {sampling_time}, to_time: {to_time}, train_time: {train_time}")
        
        return x_all


def train(model, loader, optimizer, device):
    model.train()

    total_loss = total_examples = 0
    total_batches = len(loader)
    sampling_time, to_time, train_time = 0.0, 0.0, 0.0
    
    loader_iter = iter(loader)
    while True:
        try:
            t0 = time.time()
            data = next(loader_iter)
            t1 = time.time()
            data = data.to(device)
            t2 = time.time()
            
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            y = data.y.squeeze(1)
            loss = F.nll_loss(out[data.train_mask], y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            num_examples = data.train_mask.sum().item()
            total_loss += loss.item() * num_examples
            total_examples += num_examples
            
            train_time += time.time() - t2
            to_time += t2 - t1
            sampling_time += t1 - t0   
        except StopIteration:
            break

    return total_loss / total_examples, sampling_time / total_batches, to_time / total_batches, train_time / total_batches


@torch.no_grad()
def test(model, data, evaluator, subgraph_loader, device):
    model.eval()

    out = model.inference(data.x, subgraph_loader, device)

    y_true = data.y
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[data.train_mask],
        'y_pred': y_pred[data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[data.valid_mask],
        'y_pred': y_pred[data.valid_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_mask],
        'y_pred': y_pred[data.test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc


def to_inductive(data):
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (GraphSAINT)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--inductive', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--walk_length', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_steps', type=int, default=2)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    st0 = time.time()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products', root="/home/wangzhaokang/wangyunpan/gnns-project/datasets")
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask

    # We omit normalization factors here since those are only defined for the
    # inductive learning setup.
    sampler_data = data
    if args.inductive:
        sampler_data = to_inductive(data)

    st1 = time.time()
    loader = GraphSAINTRandomWalkSampler(sampler_data,
                                         batch_size=args.batch_size,
                                         walk_length=args.walk_length,
                                         num_steps=args.num_steps,
                                         sample_coverage=0,
                                         save_dir=dataset.processed_dir)

    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=4096, shuffle=False,
                                      num_workers=12)

    st2 = time.time()
    print("batch nums: ", len(loader))
    print("sampling loader time", st2 - st1)
    
    model = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                 args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-products')
    logger = Logger(args.runs, args)
    
    st3 = time.time()
    print("preprocess time: ", st3 - st0)
    avg_sampling_time, avg_to_time, avg_train_time = 0.0, 0.0, 0.0
    avg_training_time, avg_evaluation_time = 0.0, 0.0
    
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            t0 = time.time()
            loss, sampling_time, to_time, train_time = train(model, loader, optimizer, device)
            avg_sampling_time += sampling_time
            avg_to_time += to_time
            avg_train_time += train_time
            
            t1 = time.time()
            avg_training_time += t1 - t0
            print("training total time: ", t1 - t0)
            # if epoch % args.log_steps == 0:
            #     print(f'Run: {run + 1:02d}, '
            #           f'Epoch: {epoch:02d}, '
            #           f'Loss: {loss:.4f}')

            # if epoch > 9 and epoch % args.eval_steps == 0:
            result = test(model, data, evaluator, subgraph_loader, device)
            logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * train_acc:.2f}%, '
                f'Valid: {100 * valid_acc:.2f}% '
                f'Test: {100 * test_acc:.2f}% '
                f'Time: {time.time() - t0}s')
            
            t2 = time.time()
            avg_evaluation_time += t2 - t1
            print("evaluate total time: ", t2 - t1)
            
        logger.print_statistics(run)

    avg_sampling_time /= args.runs * args.epochs
    avg_to_time /= args.runs * args.epochs
    avg_train_time /= args.runs * args.epochs
    print(f'Avg_sampling_time: {avg_sampling_time}s, '
        f'Avg_to_time: {avg_to_time}s, ',
        f'Avg_train_time: {avg_train_time}s')
    
    logger.print_statistics()
    st4 = time.time()
    print("total training time: ", st4 - st3) 
    logger.save("/home/wangzhaokang/wangyunpan/gnns-project/ogb_evaluations/exp/npy_full/products_graphsaint_sage" + str(args.batch_size))

if __name__ == "__main__":
    main()