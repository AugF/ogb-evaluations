
针对pyg-gnns中性能瓶颈分析的研究，目前拟定的motivation如下：
(1)当batch size较大时，目前的采样技术实现耗时过长。
(2)采样过程与GNN训练过程串行进行，GPU资源无法得到充分利用。
(3)Batch size的选择目前依赖人工调节，难以选择较优的batch size。

目标：现在需要根据实验验证motivation是否可靠。

以上三个motivation实际上即验证以下两个点:
- 采样耗时比例是否随batch size的增大而增大，对应于(1)和(2)
- 收敛时间和收敛精度和batch size成正相关, 对应于(3)

## 实验计划

### 数据集和算法的选取

依据："Open Graph Benchmark: Datasets for Machine Learning on Graphs
"(stanford, NIPS2020). [leader_nodeprop](https://ogb.stanford.edu/docs/leader_nodeprop/)

products: 6,7,10
cluster_gcn, graphsaint_sage, neighborsamping_gat

mag: 2,4,6
cluster_rgcn, graphsaint_rgcn, neighborsampling_rgcn

todo:

[ ] check products/neighborsampling_sage.py
[ ] run products/neighborsampling_sage.py