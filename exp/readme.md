这里为实验测试部分

epochs: 30, 30, 3; 50, 20, 100;

cluster_mag: 5000

cluster_products: 15000

实验计划

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
- [ ] 跑实验
- [ ] 开发对应的脚本分析代码