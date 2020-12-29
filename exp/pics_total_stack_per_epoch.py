
import os, sys, re 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import batches_per_epoch, val_batches_per_epoch

batch_sizes = {
    'mag_cluster_rgcn': [50, 150, 300, 500, 1000],
    'mag_graphsaint_rgcn': [1939, 9698, 19397, 58192, 116384],
    'mag_neighborsampling_rgcn': [969, 1939, 9698, 19397, 58192],
    'products_cluster_gcn': [75, 150, 450, 900, 1350],
    'products_cluster_sage': [75, 150, 450, 900, 1350],
    'products_graphsaint_sage': [244, 1224, 2449, 12245, 24490],
    'products_neighborsampling_sage': [244, 1224, 2449, 12245, 24490] 
}

relative_precent = {
    'mag_cluster_rgcn': [1, 3, 6, 10, 20],
    'mag_graphsaint_rgcn': [0.1, 0.5, 1, 3, 6],
    'mag_neighborsampling_rgcn': [0.05, 0.1, 0.5, 1, 3],
    'products_cluster_gcn': [0.5, 1, 3, 6, 9],
    'products_cluster_sage': [0.5, 1, 3, 6, 9],
    'products_graphsaint_sage': [0.01, 0.05, 0.1, 0.5, 1],
    'products_neighborsampling_sage': [0.01, 0.05, 0.1, 0.5, 1]
}

plt.style.use("ggplot")
plt.rcParams["font.size"] = 12


log_dir = "log_full"
out_dir = "res_full"  
ylabel = f"Training Time per Epoch (s)"
xlabel = f"Relative Batch Size (%)"

if not os.path.exists(out_dir + "/epoch_data"):
    os.makedirs(out_dir + "/epoch_data")

if not os.path.exists(out_dir + "/epoch_fig"):
    os.makedirs(out_dir + "/epoch_fig")
    
# mag
def run_mag():
    for file_name in batch_sizes.keys():
        if "products" in file_name:
            continue

        if "neighborsampling" in file_name:
            ns_flag = True
        else: ns_flag = False
        # 一个算法一个图像
        df_data = []
        xticklabels = [str(i) + '%' for i in relative_precent[file_name]]
        for i, bs in enumerate(batch_sizes[file_name]):
            log_path = log_dir + "/" + file_name + "_" + str(bs) + ".out"
            if not os.path.exists(log_path):
                continue
            print(log_path)
            with open(log_path) as f:
                eval_total_per_epoch, cnt = 0.0, 0
                if not ns_flag:
                    sampler_per_epoch, to_per_epoch, training_per_epoch = batches_per_epoch[file_name][i], batches_per_epoch[file_name][i], batches_per_epoch[file_name][i]
                else:
                    sampler_per_epoch, to_per_epoch, training_per_epoch = 1.0, 1.0, 1.0
                for line in f:
                    match_eval_line = re.match(r"evaluate total time:  (.*)", line)
                    match_train_line = re.match(r"Avg_sampling_time: (.*)s, Avg_to_time: (.*)s,  Avg_train_time: (.*)s", line)
                    if match_eval_line:
                        eval_total_per_epoch += float(match_eval_line.group(1))
                        cnt += 1
                    if match_train_line:
                        sampler_per_epoch *= float(match_train_line.group(1))
                        to_per_epoch *= float(match_train_line.group(2))
                        training_per_epoch *= float(match_train_line.group(3))
                eval_total_per_epoch /= cnt
            df_data.append([sampler_per_epoch, to_per_epoch, training_per_epoch, eval_total_per_epoch])
            
        # 保存数据
        df_data = np.array(df_data)    
        sampler_ratio = 100 * df_data[:, 0] / df_data.sum(axis=1)
        pd.DataFrame(df_data, index=xticklabels, columns=["Sampling", "Data Transferring", "Training", "Evaluation_total"]).to_csv(f"{out_dir}/epoch_data/" + file_name + ".csv")
        
        # 绘制图像
        colors = plt.get_cmap('Dark2')(
            np.linspace(0.15, 0.85, 5))
        
        fig, ax = plt.subplots()
        ax.bar(xticklabels, df_data[:, 3], alpha=0.6,  facecolor = colors[3], lw=1, label='Evaluation_total')
        ax.bar(xticklabels, df_data[:, 2], alpha=0.6, bottom=df_data[:, 3], facecolor = colors[2], lw=1, label='Training')
        ax.bar(xticklabels, df_data[:, 1], bottom=[df_data[i][2] + df_data[i][3] for i in range(df_data.shape[0])], alpha=0.6,  facecolor = colors[1], lw=1, label='Data Transferring')
        rects = ax.bar(xticklabels, df_data[:, 0], bottom=[df_data[i][1] + df_data[i][2] + df_data[i][3] for i in range(df_data.shape[0])], alpha=0.6,  facecolor = colors[0],  lw=1, label='Sampling')
        
        for i, rect in enumerate(rects):
            ax.text(rect.get_x() + rect.get_width() / 3, rect.get_y() + rect.get_height() / 3, '%.1f' % sampler_ratio[i])

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend()#图例展示位置，数字代表第几象限
        fig.savefig(f"{out_dir}/epoch_fig/" + file_name + ".png")
    

# products 
def run_products():
    for file_name in batch_sizes.keys():
        if "mag" in file_name:
            continue
        if "graphsaint" in file_name:
            gs_flag = True
        else: gs_flag = False
        # 一个算法一个图像
        df_data = []
        xticklabels = [str(i) + '%' for i in relative_precent[file_name]]
        columns = ["Sampler_train", "Sampler_eval", "To_train", "To_eval", "Training", "Training_eval"]
        val_batches = val_batches_per_epoch[file_name]
        for i, bs in enumerate(batch_sizes[file_name]):
            log_path = log_dir + "/" + file_name + "_" + str(bs) + ".out"
            if not os.path.exists(log_path):
                continue
            print(log_path)
            total_batches = batches_per_epoch[file_name][i]
            with open(log_path) as f:
                sampler_eval_per_epoch, to_eval_per_epoch, training_eval_per_epoch, cnt = 0.0, 0.0, 0.0, 0
                if not gs_flag:
                    sampler_per_epoch, to_per_epoch, training_per_epoch = total_batches, total_batches, total_batches
                else:
                    sampler_per_epoch, to_per_epoch, training_per_epoch = 1.0, 1.0, 1.0
                for line in f:
                    match_eval_line = re.match(r"Evaluation: sampling time: (.*), to_time: (.*), train_time: (.*)", line)
                    match_train_line = re.match(r"Avg_sampling_time: (.*)s, Avg_to_time: (.*)s,  Avg_train_time: (.*)s", line)
                    if match_eval_line:
                        sampler_eval_per_epoch += float(match_eval_line.group(1)) * val_batches
                        to_eval_per_epoch += float(match_eval_line.group(2)) * val_batches
                        training_eval_per_epoch += float(match_eval_line.group(3)) * val_batches
                        cnt += 1
                    if match_train_line:
                        sampler_per_epoch *= float(match_train_line.group(1))
                        to_per_epoch *= float(match_train_line.group(2))
                        training_per_epoch *= float(match_train_line.group(3))
                sampler_eval_per_epoch /= cnt
                to_eval_per_epoch /= cnt
                training_eval_per_epoch /= cnt
                
            df_data.append([sampler_per_epoch, sampler_eval_per_epoch, to_per_epoch, to_eval_per_epoch, training_per_epoch, training_eval_per_epoch])
            
        # 保存数据
        df_data = np.array(df_data)    
        sampler_ratio = 100 * df_data[:, 0] / df_data.sum(axis=1)
        sampler_eval_ratio = 100 * df_data[:, 1] / df_data.sum(axis=1)
        pd.DataFrame(df_data, index=xticklabels, columns=columns).to_csv(f"{out_dir}/epoch_data/" + file_name + ".csv")
        
        # 绘制图像
        colors = plt.get_cmap('Dark2')(
            np.linspace(0.15, 0.85, 6))
        
        fig, ax = plt.subplots()
        ax.bar(xticklabels, df_data[:, 5], alpha=0.6,  facecolor = colors[5], lw=1, label=columns[5])
        ax.bar(xticklabels, df_data[:, 4], alpha=0.6, bottom=df_data[:, 5], facecolor = colors[4], lw=1, label=columns[4])
        ax.bar(xticklabels, df_data[:, 3], bottom=[df_data[i][4] + df_data[i][5] for i in range(df_data.shape[0])], alpha=0.6,  facecolor = colors[3], lw=1, label=columns[3])
        ax.bar(xticklabels, df_data[:, 2], bottom=[df_data[i][3] + df_data[i][4] + df_data[i][5] for i in range(df_data.shape[0])], alpha=0.6,  facecolor = colors[2], lw=1, label=columns[2])
        rects_eval = ax.bar(xticklabels, df_data[:, 1], bottom=[df_data[i][2] + df_data[i][3] + df_data[i][4] + df_data[i][5] for i in range(df_data.shape[0])], alpha=0.6,  facecolor = colors[1],  lw=1, label=columns[1])
        rects = ax.bar(xticklabels, df_data[:, 0], bottom=[df_data[i][1] + df_data[i][2] + df_data[i][3] + df_data[i][4] + df_data[i][5] for i in range(df_data.shape[0])], alpha=0.6,  facecolor = colors[0],  lw=1, label=columns[0])
        
        for i, rect in enumerate(rects):
            ax.text(rect.get_x() + rect.get_width() / 3, rect.get_y() + rect.get_height() / 3, '%.1f' % sampler_ratio[i])

        # for i, rect in enumerate(rects_eval):
        #     ax.text(rect.get_x() + rect.get_width() / 3, rect.get_y() + rect.get_height() / 3, '%.1f' % sampler_eval_ratio[i])
            
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend()#图例展示位置，数字代表第几象限
        fig.savefig(f"{out_dir}/epoch_fig/" + file_name + ".png")


run_mag()
run_products()