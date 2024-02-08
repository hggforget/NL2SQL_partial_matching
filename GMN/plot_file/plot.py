import numpy as np
import matplotlib.pyplot as plt


# EMA函数
def exponential_moving_average(data, alpha):
    ema = [data[0]]  # 初始值为第一个数据点
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return np.array(ema)


# 设置EMA的alpha参数（平滑程度可以调整，通常在0到1之间）
alpha = 0.05


def extract_diff_type(file_path):
    output_file_list = []

    # 读取文本文件
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 初始化一个字典，用于按照类型存储行
    iter_dict = {}

    # 遍历每一行，根据类型将行添加到相应的列表中
    for line in lines:
        parts = line.split(', ')
        iter_type = parts[-1].strip()  # 获取类型

        if "dev" not in iter_type:
            return [file_path]

        if iter_type not in iter_dict:
            iter_dict[iter_type] = []
        iter_dict[iter_type].append(line)

    # 遍历字典，将每个类型的行写入相应的文件
    for iter_type, iter_lines in iter_dict.items():
        ori_file_name = file_path.replace(".txt", "")
        filename = f'{ori_file_name}_{iter_type}.txt'
        output_file_list.append(filename)

        with open(filename, 'w') as output_file:
            output_file.writelines(iter_lines)

    return output_file_list



def extract_values_from_file(filename):
    iter_values = []
    auc_values = []

    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            for part in parts:
                if 'val/pair_auc' in part:
                    auc = float(part.split()[-1])
                    auc_values.append(auc)
                elif 'iter' in part:
                    iter_num = int(part.split()[-1])
                    iter_values.append(iter_num)

    # 计算EMA
    ema_auc_values = exponential_moving_average(auc_values, alpha)

    return iter_values, auc_values, ema_auc_values


def plot_data(iter_values, auc_values, ema_auc_values, plot_lines=True, plot_ema=True, label=None):
    if plot_lines:
        plt.plot(iter_values, auc_values, marker='o', linestyle='-', linewidth=1, markersize=2, alpha=0.2)

    if plot_ema:
        plt.plot(iter_values, ema_auc_values, linestyle='--', label=f'EMA ({label})')


# 文件名数组
# log_files = ["plot_margin_0.2.txt", "plot_margin_0.5.txt", "plot_margin_0.8.txt", "plot_data_edge_margin1.0_conv.txt", "plot_margin_1.2.txt", "plot_margin_1.5.txt"]
# log_files = ["plot_data_edge_margin1.0_conv.txt", "plot_aug_balance.txt", "plot_augmentation.txt"]
# log_files = ["plot_data_edge_margin1.0_conv.txt", "plot_resnet.txt"]
# log_files = ["plot_margin_0.4_resnet_aug1400.txt", "plot_resnet_margin0.5_aug1200.txt", "plot_resnet_margin0.5_aug1200_nodata.txt", "plot_resnet_margin0.5_aug1400.txt"]
# log_files = ["plot_renset_margin0.4_hash_aug1400.txt", "plot_margin_0.4_resnet_aug1400.txt", "plot_resnet_margin0.4_init.txt"]
# log_files = [
#     'plot_relnode_seperated.txt',
#     'plot_relnode.txt',
#     'plot_ast.txt',
#     'plot_relnode_pure.txt'
# ]
# log_files_all = []
# for filename in log_files:
#     new_files = extract_diff_type(filename)
#     log_files_all = log_files_all + new_files
#
# log_files_all = [
#     "plot_ast_dev_dataset.txt",
#     "plot_relnode_dev_dataset.txt",
#     "plot_relnode_pure_dev_dataset.txt",
#     "plot_relnode_separated_dev_dataset.txt",
#     # "plot_relnode_whole_dev_dataset.txt",
#     "plot_rw_cross_attention_randombatch_d_dev_dataset.txt",
#     # "same_subtree_ast_dev_dataset.txt"
# ]

log_files_all = [
    # "plot_ast_dev_gpt4_dataset.txt",
    "plot_relnode_dev_gpt4_dataset.txt",
    "plot_relnode_pure_dev_gpt4_dataset.txt",
    "plot_relnode_separated_dev_gpt4_dataset.txt",
    # "plot_relnode_whole_dev_gpt4_dataset.txt",
    "plot_rw_cross_attention_randombatch_d_dev_gpt4_dataset.txt",
    # "same_subtree_ast_dev_gpt4_dataset.txt"
]
# 创建一个图形窗口
plt.figure(figsize=(18, 10))

# 处理每个文件并绘制图表
for filename in log_files_all:
    iter_values, auc_values, ema_auc_values = extract_values_from_file(filename)
    plot_data(iter_values, auc_values, ema_auc_values, plot_lines=True, plot_ema=True,
              label=filename.replace("log_", "").replace(".txt", ""))

plt.xlabel('Epochs')
plt.ylabel('Val/Pair_AUC')
plt.title('Val/Pair_AUC vs. Epochs with EMA')
plt.grid(True)
plt.legend()
plt.show()
