这里为实验测试部分

epochs: 30, 30, 3; 50, 20, 100;

cluster_mag: 5000

cluster_products: 15000

## 实验计划

- [x] 确定选择哪些算法，mag和products各选3个代表性算法
- [x] 修改实验代码，可以输入备选参数
- [x] 代码正确性测试
- [x] 设计batch_size
    - mag_cluster_rgcn:
        > num_partitions=5000, batch_size=500 (10%); runs=10, epochs=30; 50
        > < 50%
        >  1%, 3%, 6%, 10%, 20%
    - mag_graphsaint_rgcn: 
        > nodes: 1939743, batch_size=20000 (0.02); runs=10, epochs=30; 50
        > < 10%
        > 0.1%, 0.5%, 1%, 3%, 6%
    - mag_neighborsampling_rgcn:
        > nodes: 1939743, batch_size=1024 (0.0005); runs=10, epochs=3;  20
        > < 6%
        > 0.05%, 0.1%, 0.5%, 1%, 3%
    - products_cluster_gcn:
        > num_partitions=15000, batch_size=32 (0.002); runs=10, epochs=50; 75
        > < 50%
        > 0.5%, 1%, 3%, 6%, 9%
    - products_cluster_sage
        > num_partitions=15000, batch_size=32 (0.002); runs=10, epochs=50; 75
        > < 50%
        > 0.5%, 1%, 3%, 6%, 10%
    - products_graphsaint_sage
        > nodes: 2449029, batch_size=20000 (0.008); runs=10, epochs=20; 40
        > < 2%
        > 0.01%, 0.05%, 0.1%, 0.5%, 1%
- [x] 跑实验
- [x] 开发对应的脚本分析代码

## 新实验计划
> 该组实验中不包含 products_neighborsampling_sage
> 实验结果在log_full, npy_full文件中

- [x] 开发代码
- [x] 跑实验
- [ ] check代码是否正确
    - [x] check Avg Batch的代码收集部分
        > (暂时不收集products_cluster_sage的结果, 未跑完)
        > mag_neighborsampling_rgcn, products_graphsaint_sage忘记除以total_batches, 进行补救; 
        > log_full, npy_full是旧版的结果，代码已经更正
    - [x] check Sample的耗时比例
        > Sampler_training, Sampler_evaluation, Data Transferring, Training
        > 注意到这里的单位应该是epoch
        - mag: 无Sampler_evaluation部分, sampler_training * len(train_loader), evaluation_total
            - cluster_rgcn: 
            - graphsaint_rgcn: 
            - neighborsampling_rgcn: 
        - products: sampler_training, sampler_evaluation, to_training, to_evaluation, train_training, train_evaluation
            - cluster_gcn, cluster_sage: 2392
            - graphsaint_sage, neighborsampling_sage: 598