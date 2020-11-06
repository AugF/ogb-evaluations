
batch_sizes = {
    'mag_cluster_rgcn': [50, 150, 300, 500, 1000],
    'mag_graphsaint_rgcn': [1939, 9698, 19397, 58192, 116384],
    'mag_neighborsampling_rgcn': [969, 1939, 9698, 19397, 58192],
    'products_cluster_gcn': [75, 150, 450, 900, 1350],
    'products_cluster_sage': [75, 150, 450, 900, 1500],
    'products_graphsaint_sage': [244, 1224, 2449, 12245, 24490],
    'products_neighborsampling_sage': [244, 1224, 2449, 12245, 24490]
}

relative_precent = {
    'mag_cluster_rgcn': [1, 3, 6, 10, 20],
    'mag_graphsaint_rgcn': [0.1, 0.5, 1, 3, 6],
    'mag_neighborsampling_rgcn': [0.05, 0.1, 0.5, 1, 3],
    'products_cluster_gcn': [0.5, 1, 3, 6, 9],
    'products_cluster_sage': [0.5, 1, 3, 6, 10],
    'products_graphsaint_sage': [0.01, 0.05, 0.1, 0.5, 1],
    'products_neighborsampling_sage': [0.01, 0.05, 0.1, 0.5, 1]
}

batches_per_epoch = {
    'mag_cluster_rgcn': [100, 34, 17, 10, 5],
    'mag_graphsaint_rgcn': [30, 30, 30, 30, 30],
    'mag_neighborsampling_rgcn': [650, 325, 65, 33, 11],
    'products_cluster_gcn': [200, 100, 34, 17, 12],
    'products_cluster_sage': [200, 100, 34, 17, 12],
    'products_graphsaint_sage': [30, 30, 30, 30, 30],
    'products_neighborsampling_sage': [806, 161, 81, 17, 9]
}

