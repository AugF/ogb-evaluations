这里为https://ogb.stanford.edu/docs/leader_nodeprop/中所引用的算法

# Motivation验证性实验

## 实验目标

验证以下三个优化点：
1. 当batch_size较大时，目前的采样技术实现耗时占比过大
2. 采样技术和GNN训练过程串行进行，GPU资源无法得到充分利用
3. Batch Size的选择需要人工调节，难以选择最优的batch size

拟通过实验，发现下面的问题：
1. 采样耗时比例随着Batch Size的变化
2. 训练时间和训练精度随着Batch Size的变化

## 实验思路

### 已完成
- [x] 调研已有的采样算法
- [x] 寻找现有的高精度的GNN算法的baselines
- [x] 对训练阶段过程的性能瓶颈进行发掘

### 下一步计划
- [ ] 收集neighborsampling相关的数据
- [ ] 确定实际训练中的耗时比例
- [ ] 确定不同的sampler算法所占的比例

## 实验计划

- [ ] 统计训练过程

总运行时间：
- preprocess time
    - sampling loader time
- total training time
    - training_total_time
    - evaluation total time