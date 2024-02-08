from GMN.handle_dataset.spider_dataset import SpiderDataset
from GMN.handle_dataset.data_pre_processor.ast_processor import ASTProcessor
from GMN.handle_dataset.data_pre_processor.positional_encoding_processor import PositionalEncodingProcessor
from GMN.evaluation import compute_similarity, auc
from GMN.loss import pairwise_loss
from utils import *
from GMN.configure import *
import numpy as np
import torch.nn as nn
import time
import os
import random
import pandas as pd


def model_eval_dev(dev_dataset : SpiderDataset, name):

    with (torch.no_grad()):

        similarity_array = torch.empty(0).to(device)
        labels_array = torch.empty(0).to(device)
        dev_dataset.shuffle()
        for edge_tuple, node_tuple, n_graphs, batch_labels in dev_dataset:
            labels = batch_labels.to(device)
            eval_pairs = model(edge_tuple.to(device), node_tuple.to(device), n_graphs)

            x, y = reshape_and_split_tensor(eval_pairs,  int(n_graphs / 2))
            similarity = compute_similarity(config, x, y).to(device)

            similarity_array = torch.cat((similarity_array, similarity), dim=0)
            labels_array = torch.cat((labels_array, labels), dim=0)

        pair_auc = auc(similarity_array, labels_array)
        log_file_txt = 'iter %d, loss %.4f, val/pair_auc %.6f, time %.2fs, %s\n' % (i_epoch, loss_mean.item(), pair_auc, time.time() - t_start, name)

        print('iter %d, loss %.4f, val/pair_auc %.4f, time %.2fs, %s' %
              (i_epoch, loss_mean.item(), pair_auc, time.time() - t_start, name))

        return log_file_txt


is_cuda = torch.cuda.is_available()
ONLINE_TRAIN_SETTINGS = {
    'NAME': "no_ascii_match_data_type",
    'GPU_DEVICE': torch.device('cuda:0' if is_cuda else 'cpu'),
    'PATH_TO_GMN': "/opt/tiger/NL2SQL_partial_matching/GMN",
    'IMAGES_FOLDER': "save_file/picture/",
    'LOG_FILE_NAME': "save_file/log/plot",
    'Checkpoint': 'save_file/checkpoints/',

    'WH_train': 'database_spider/gmn_api_data/wh_in_sample_train.xlsx',
    'WH_dev': 'database_spider/gmn_api_data/wh_in_sample_dev.xlsx',

    'Train': 'database_spider/gmn_api_data/train.xlsx',
    'Dev': 'database_spider/gmn_api_data/test.xlsx',
    'Dev_gpt4': 'database_spider/gmn_api_data/test_aug.xlsx',
}

print("==================================")
device = ONLINE_TRAIN_SETTINGS['GPU_DEVICE']
name = ONLINE_TRAIN_SETTINGS['NAME']
print("Device:", device)
print("Name:", name)

if is_cuda:
    image_folder_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['IMAGES_FOLDER'])
    log_folder_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['LOG_FILE_NAME'])
    checkpoint_folder_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['Checkpoint'])

    train_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['Train'])
    dev_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['Dev'])

    wh_train_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['WH_train'])
    wh_dev_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['WH_dev'])
    dev_gpt4_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['Dev_gpt4'])

    print("Images saved in:", image_folder_path)
    print("Logs saved in:", log_folder_path)
    print("==================================")

else:
    image_folder_path = ONLINE_TRAIN_SETTINGS['IMAGES_FOLDER']
    log_folder_path = ONLINE_TRAIN_SETTINGS['LOG_FILE_NAME']
    checkpoint_folder_path = ONLINE_TRAIN_SETTINGS['Checkpoint']

    train_path = ONLINE_TRAIN_SETTINGS['Train']
    dev_path = ONLINE_TRAIN_SETTINGS['Dev']

    wh_train_path = ONLINE_TRAIN_SETTINGS['WH_train']
    wh_dev_path = ONLINE_TRAIN_SETTINGS['WH_dev']
    dev_gpt4_path = ONLINE_TRAIN_SETTINGS['Dev_gpt4']


torch.set_default_tensor_type(torch.FloatTensor)
# Print configure
config = get_default_config()
for (k, v) in config.items():
    print("%s= %s" % (k, v))

# Set random seeds
seed = config['seed']
np.random.seed(seed)
np.random.seed(seed + 1)
random.seed(seed)
torch.manual_seed(seed + 2)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


train_df = pd.read_excel(train_path, sheet_name="Sheet1").reset_index(drop=True)
dev_df = pd.read_excel(dev_path, sheet_name="Sheet1").reset_index(drop=True)
#
# wh_train_df = clean_df(pd.read_excel(wh_train_path, sheet_name="Sheet1")).reset_index(drop=True)
# wh_dev_df = clean_df(pd.read_excel(wh_dev_path, sheet_name="Sheet1")).reset_index(drop=True)
#
dev_gpt4_df = pd.read_excel(dev_gpt4_path, sheet_name="Sheet1").reset_index(drop=True)
#
# con_train_df = pd.concat([train_df, wh_train_df], axis=0).reset_index(drop=True)

nums = 30

train_df = train_df.head(nums)
dev_df = dev_df.head(nums)
dev_gpt4_df = dev_gpt4_df.head(nums)


data_pre_processor = ASTProcessor()
pair_list_train, labels_train = data_pre_processor.read_data(train_df)
train_batch_data = data_pre_processor.pairs_spider(config['training']['batch_size'], pair_list_train, labels_train)

pair_list_dev, labels_dev = data_pre_processor.read_data(dev_df)
dev_batch_data = data_pre_processor.pairs_spider(config['training']['batch_size'], pair_list_dev, labels_dev)

pair_list_dev_gpt4, labels_dev_gpt4 = data_pre_processor.read_data(dev_gpt4_df)
dev_gpt4_batch_data = data_pre_processor.pairs_spider(config['training']['batch_size'], pair_list_dev_gpt4, labels_dev_gpt4)


# from evaluation import auc_roc
# auc_roc(train_df)
# auc_roc(dev_df)

# train_df.to_excel("wh_train.xlsx", "Sheet1")
# dev_df.to_excel("wh_dev.xlsx", "Sheet1")


model, optimizer = build_model(config)
model.to(device)

training_dataset = SpiderDataset(train_batch_data)

dev_dataset = SpiderDataset(dev_batch_data)
# wh_dev_dataset = SpiderDataset(wh_dev_batch_data)
dev_gpt4_dataset = SpiderDataset(dev_gpt4_batch_data)

log_txt = ""

for i_epoch in range(config['training']['n_training_steps']):
    model.train(True)
    t_start = time.time()

    loss_mean = torch.empty(0).to(device)
    training_dataset.shuffle()
    for edge_tuple, node_tuple, n_graphs, batch_labels in training_dataset:
        labels = batch_labels.to(device)
        graph_vectors = model(edge_tuple.to(device), node_tuple.to(device), n_graphs)

        x, y = reshape_and_split_tensor(graph_vectors,  int(n_graphs / 2))
        loss = pairwise_loss(x, y, labels, loss_type=config['training']['loss'], margin= config['training']['margin'])
        loss_mean = torch.mean(loss)

        optimizer.zero_grad()
        # add
        loss.backward(torch.ones_like(loss))
        nn.utils.clip_grad_value_(model.parameters(), config['training']['clip_value'])
        optimizer.step()

    model.eval()

    dev_log = model_eval_dev(dev_dataset, "dev_dataset")
    # wh_dev_log = model_eval_dev(wh_dev_dataset, "wh_dev_dataset")
    dev_gpt4_log = model_eval_dev(dev_gpt4_dataset, "dev_gpt4_dataset")

    log_txt = log_txt + dev_log + dev_gpt4_log

    if i_epoch % 2 == 0 and i_epoch != 0:
        checkpoint_save_path = f"{checkpoint_folder_path}{name}_{i_epoch}.pt"
        torch.save(model.state_dict(), checkpoint_save_path)
        log_file_path = f"{log_folder_path}_{name}.txt"

        with open(log_file_path, 'a') as log_file:
            log_file.write(log_txt)
            log_file.flush()
            log_txt = ""










