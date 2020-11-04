'''
该文件是为了画train, val, test随epoch变化的图像
'''
import os
import re
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

batch_sizes = {
    'mag_cluster_rgcn': [50, 150, 300, 500, 1000],
    'mag_graphsaint_rgcn': [1939, 9698, 19397, 58192, 116384],
    'mag_neighborsampling_rgcn': [969, 1939, 9698, 19397, 58192],
    'products_cluster_gcn': [75, 150, 450, 900, 1350],
    'products_cluster_sage': [75, 150, 450, 900, 1500],
    'products_graphsaint_sage': [244, 1224, 2449, 12245, 24490]
}

total_nums = {
    'mag_cluster_rgcn': 5000,
    'mag_graphsaint_rgcn': 1939743,
    'mag_neighborsampling_rgcn': 1939743,
    'products_cluster_gcn': 15000,
    'products_cluster_sage': 15000,
    'products_graphsaint_sage': 2449029
}

relative_precent = {
    'mag_cluster_rgcn': [1, 3, 6, 10, 20],
    'mag_graphsaint_rgcn': [0.1, 0.5, 1, 3, 6],
    'mag_neighborsampling_rgcn': [0.05, 0.1, 0.5, 1, 3],
    'products_cluster_gcn': [0.5, 1, 3, 6, 9],
    'products_cluster_sage': [0.5, 1, 3, 6, 10],
    'products_graphsaint_sage': [0.01, 0.05, 0.1, 0.5, 1]
}


os.chdir("/home/wangzhaokang/wangyunpan/gnns-project/ogb_evaluations/exp")

def handle(alg, para):
    para = str(para)
    file = "npy/" + alg + para + ".npy"
    if not os.path.exists(file):
        return
    x = np.load(file)
    df = pd.DataFrame(x[0], columns=['train', 'val', 'test'])
    fig, ax = plt.subplots()
    plt.ylim(0, 1)
    df.plot(ax=ax, kind='line')
    fig.savefig("res/" + alg + para + ".png")
    plt.close()


def pics_train():
    for file_name in batch_sizes.keys():
        for para in batch_sizes[file_name]:
            handle(alg=file_name, para=para)

def save_acc():
    acc = {}
    for file_name in batch_sizes.keys():
        acc[file_name] = []
        for bs in batch_sizes[file_name]:
            log_path = "log/" + file_name + "_" + str(bs) + ".out"
            with open(log_path) as f:
                for line in f:
                    match_line = re.match(r".*Final Test: (.*)", line)
                    if match_line:
                        acc[file_name].append(float(match_line.group(1)))
                        break
    np.save("acc", acc)
    
    
def get_max_epochs_acc():
    for file_name in batch_sizes.keys():
        print(file_name)
        df_data = {}
        for bs in batch_sizes[file_name]:
            # get avg_batch_time
            log_path = "log/" + file_name + '_' + str(bs) + ".out"
            train_time, sampling_time, to_time = 0.0, 0.0, 0.0
            with open(log_path) as f:
                for line in f:
                    match_line = re.match(r"Avg_sampling_time: (.*)s, Avg_to_time: (.*)s,  Avg_train_time: (.*)s", line)
                    if match_line:
                        sampling_time = float(match_line.group(1))
                        to_time = float(match_line.group(2))
                        train_time = float(match_line.group(3))
                        success = True
                        break
            avg_batch_time = sampling_time + to_time + train_time
            
            # get acc, epoch
            npy_path = "npy/" + file_name + str(bs) + ".npy"
            batch_size = int(total_nums[file_name] / bs) + 1
            print(bs, "batch nums", batch_size)
            results = np.load(npy_path)
            result = 100 * torch.tensor(results)

            best_results = []
            final_epoch = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                final_epoch.append(r[:, 1].argmax())
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)
            # 固定轮数取max
            df_data[bs] = [final_epoch[0].item(), batch_size * final_epoch[0].item(), avg_batch_time, batch_size * final_epoch[0].item() * avg_batch_time, best_result[:, 3].item()]
            # print("epochs:", final_epoch[0].item(), "batchs: ", batch_size * final_epoch[0].item(), ", acc: ", best_result[:, 3].item())
        df = pd.DataFrame(df_data, index=['epochs', 'batchs', 'avg_batch_time', 'total time', 'acc']).T
        df.index = [str(i) + "%" for i in relative_precent[file_name]]
        df.to_csv("res/acc_epoch/" + file_name + ".csv")
        # fig, ax = plt.subplots()
        # df['epochs'].plot(ax=ax, marker='.')
        # df['acc'].plot(ax=ax, marker='.')
        # fig.savefig("res/acc_epoch/" + file_name + ".png")


def get_max_epochs_acc():
    for file_name in batch_sizes.keys():
        print(file_name)
        df_data = {}
        for bs in batch_sizes[file_name]:
            # get avg_batch_time
            log_path = "log/" + file_name + '_' + str(bs) + ".out"
            train_time, sampling_time, to_time = 0.0, 0.0, 0.0
            with open(log_path) as f:
                for line in f:
                    match_line = re.match(r"Avg_sampling_time: (.*)s, Avg_to_time: (.*)s,  Avg_train_time: (.*)s", line)
                    if match_line:
                        sampling_time = float(match_line.group(1))
                        to_time = float(match_line.group(2))
                        train_time = float(match_line.group(3))
                        success = True
                        break
            avg_batch_time = sampling_time + to_time + train_time

get_max_epochs_acc()