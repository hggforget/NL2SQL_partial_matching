import numpy
from matplotlib import pyplot as plt
import threading
from GMN.handle_dataset.spider_dataset import SpiderDataset
from queue import Queue,LifoQueue,PriorityQueue
from GMN.evaluation import compute_similarity, auc
from GMN.utils import *
from GMN.configure import *
import os
from GMN.handle_dataset.data_pre_processor.positional_encoding_processor import PositionalEncodingProcessor
import pandas as pd
import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
import concurrent.futures
import scipy.stats as stats


dir_path = os.path.split(os.path.realpath(__file__))[0]
import time


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 的运行时间为：{end_time - start_time} 秒")
        return result
    return wrapper


def init():
    torch.set_default_tensor_type(torch.FloatTensor)
    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Print configure
    config = get_default_config()
    for (k, v) in config.items():
        print("%s= %s" % (k, v))

    # Set random seeds
    seed = config['seed']
    np.random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    return device, config


def load_data_from_excel(excel_path="database_spider/dev.xlsx"):
    dev_df = pd.read_excel(excel_path, sheet_name="Sheet1")[:300]
    data_processor = PositionalEncodingProcessor()
    pairs, labels = data_processor.read_data(dev_df)

    return pairs, labels, dev_df


def generate_path(model_name='cross_attention_positional_whole_graph.pt'):
    checkpoints_dir_relative_path = '../save_file/checkpoints/'
    checkpoints_dir_path = os.path.join(dir_path, checkpoints_dir_relative_path)
    return os.path.abspath(os.path.join(checkpoints_dir_path, model_name))


def load_model_by_checkpoint(model_path, config, device):
    try:
        # 从 checkpoint 中加载模型参数
        checkpoint = torch.load(model_path, map_location=device)
        model, optimizer = build_model(config)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
    except Exception as e:
        raise Exception("Error occurred when loading checkpoint\n", str(e))
    return model


@timer_decorator
def model_eval(batch_data):
    model, device, config, edge_tuple, node_tuple, n_graphs = batch_data
    # print("==================== Model Prediction ====================")
    eval_pairs = model(edge_tuple.to(device), node_tuple.to(device), n_graphs)

    x, y = reshape_and_split_tensor(eval_pairs, config['training']['batch_size'])
    similarity = compute_similarity(config, x, y).to(device)

    return similarity


def save_to_excel(similarity, labels, model_path, dev_df):


    fpr, tpr, thresholds = metrics.roc_curve(labels.cpu().detach().numpy(), similarity.cpu().detach().numpy())

    print("scores", similarity, len(labels), thresholds)

    # 初始化变量以保存最佳阈值和最高准确率
    best_threshold = None
    best_accuracy = 0.0

    # 遍历所有阈值，并计算准确率
    for threshold in thresholds:
        # 使用当前阈值将 scores 转化为二进制分类结果
        predicted_labels = (similarity.cpu().detach().numpy() > threshold).astype(int)

        # 计算准确率
        accuracy = accuracy_score(labels.cpu().detach().numpy(), predicted_labels)

        # 如果当前准确率更高，则更新最佳阈值和最高准确率
        if accuracy > best_accuracy:
            best_threshold = threshold
            best_accuracy = accuracy

    print(f'Best Threshold: {best_threshold}')
    print(f'Best Accuracy: {best_accuracy:.2f}')

    model_path_name = (model_path.split("/")[-1]).replace(".pt", "")
    excel_file_name = f'predictions/{model_path_name}_{best_threshold}_{best_accuracy:.2f}.xlsx'

    # 创建一个 DataFrame，将列表数据添加为列
    data = {'Predict Scores': similarity.detach().numpy(), 'Labels': labels.detach().numpy(),
            "threshold": np.array([best_threshold for i in range(len(labels))])}
    df = pd.DataFrame(data)
    result_df = pd.concat([dev_df, df], axis=1)
    result_df.to_excel(excel_file_name, index=False)


model_queue = Queue()


# def get_model():
#     global model_queue
#     if not model_queue.empty():
#         return model_queue.get()
#     else:
#         time.sleep(1)
#         return get_model()


def get_similarity(row):

    g_rel_node, p_rel_node, db_id = row[10], row[11], row[6]
    prediction = model_eval(model=model,
                            device=device,
                            config=config,
                            g_sql_rel=g_rel_node,
                            p_sql_rel=p_rel_node,
                            db_id=db_id)
    return prediction


def get_threshold(label_list, similarity_list):
    fpr, tpr, thresholds = metrics.roc_curve(label_list, similarity_list)

    # 初始化变量以保存最佳阈值和最高准确率
    best_threshold = None
    best_accuracy = 0.0

    # 遍历所有阈值，并计算准确率
    for threshold in thresholds:
        # 使用当前阈值将 scores 转化为二进制分类结果
        predicted_labels = (similarity_list > threshold).astype(int)

        # 计算准确率
        accuracy = accuracy_score(label_list, predicted_labels)

        # 如果当前准确率更高，则更新最佳阈值和最高准确率
        if accuracy > best_accuracy:
            best_threshold = threshold
            best_accuracy = accuracy

    print(f'Best Threshold: {best_threshold}')
    print(f'Best Accuracy: {best_accuracy:.2f}')

    threshold_list = [best_threshold for i in range(len(similarity_list))]


def get_GMN_score_df():
    # pairs, labels, df = load_data_from_excel("../database_spider/gmn_api_data/dev_all_fix_gpt4.xlsx")
    df = pd.read_excel("../database_spider/gmn_api_data/test_aug.xlsx", sheet_name="Sheet1")\
        .reset_index(drop=True)
    data_pre_processor = PositionalEncodingProcessor()
    pair_list_train, labels_train = data_pre_processor.read_data(df)
    train_batch_data = data_pre_processor.pairs_spider(config['training']['batch_size'], pair_list_train, labels_train)
    training_dataset = SpiderDataset(train_batch_data)
    batch_list = [(model, device, config, edge_tuple, node_tuple, n_graphs) for
                  edge_tuple, node_tuple, n_graphs, batch_labels in training_dataset]
    # 创建ThreadPoolExecutor，指定线程数量
    start = time.time()
    num_threads = 5  # 你可以根据需要设置线程数量
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 并行执行API请求
        results = list(executor.map(model_eval, batch_list))
    labels_array = torch.cat([batch_labels for
                             edge_tuple, node_tuple, n_graphs, batch_labels in training_dataset], dim=0)
    similarity_array = torch.cat(results, dim=0)
    # df = pd.concat([df, result_df], axis=1)
    # print(df)
    pair_auc = auc(similarity_array, labels_array)
    kendall, _ = stats.kendalltau(similarity_array.tolist(), labels_array.tolist())
    spearman, _ = stats.spearmanr(similarity_array.tolist(), labels_array.tolist())

    end = time.time()
    print("Time Spend", (end - start) / similarity_array.shape[0])
    print("auc ", pair_auc)


if __name__ == '__main__':
    device, config = init()
    model_path = generate_path('ast_600.pt')
    model = load_model_by_checkpoint(model_path=model_path, config=config, device=device)

    # g_rel_node = """{"rels": [{"id": "0","operator": "TableScan","table": ["HIVE_BOE","spider_course_teach","teacher"],"extraDigest": "StatsTable","inputs": [],"outputs": [["teacher_id","INTEGER"],["name","STRING"],["age","STRING"],["hometown","STRING"]]},{"id": "1","operator": "Project","outputs": [["teacher_id","INTEGER"],["name","STRING"]],"exprs": [{"input": 0},{"input": 1}],"inputs": ["0"]},{"id": "2","operator": "TableScan","table": ["HIVE_BOE","spider_course_teach","course_arrange"],"extraDigest": "StatsTable","inputs": [],"outputs": [["course_id","INTEGER"],["teacher_id","INTEGER"],["grade","INTEGER"]]},{"id": "3","operator": "Project","outputs": [["teacher_id","INTEGER"]],"exprs": [{"input": 1}],"inputs": ["2"]},{"id": "4","operator": "HashJoin","condition": {"op": {"name": "=","kind": "EQUALS"},"operands": [{"input": 0},{"input": 2}]},"joinType": "inner","inputs": ["1","3"],"outputs": [["teacher_id","INTEGER"],["name","STRING"],["teacher_id0","INTEGER"]]},{"id": "5","operator": "HashAggregate","group": [1],"aggs": [{"agg": {"name": "COUNT","kind": "COUNT"},"type": {"type": "Int64","nullable": false},"distinct": false,"filter": -1,"approximate": false,"ignoreNulls": false,"operands": [],"name": null}],"mode": "None","inputs": ["4"],"outputs": [["name","STRING"],["$f1","BIGINT"]]},{"id": "6","operator": "Filter","condition": {"op": {"name": ">=","kind": "GREATER_THAN_OR_EQUAL"},"operands": [{"input": 1},{"literal": 2,"type": {"type": "Int64","nullable": false}}]},"inputs": ["5"],"outputs": [["name","STRING"],["$f1","BIGINT"]]},{"id": "7","operator": "Project","outputs": [["name","STRING"]],"exprs": [{"input": 0}],"inputs": ["6"]}]}"""
    # p_rel_node = """{"rels": [{"id": "0","operator": "TableScan","table": ["HIVE_BOE","spider_course_teach","teacher"],"extraDigest": "StatsTable","inputs": [],"outputs": [["teacher_id","INTEGER"],["name","STRING"],["age","STRING"],["hometown","STRING"]]},{"id": "1","operator": "Project","outputs": [["teacher_id","INTEGER"],["name","STRING"]],"exprs": [{"input": 0},{"input": 1}],"inputs": ["0"]},{"id": "2","operator": "TableScan","table": ["HIVE_BOE","spider_course_teach","course_arrange"],"extraDigest": "PartitionedTable_List()","inputs": [],"outputs": [["course_id","INTEGER"],["teacher_id","INTEGER"],["grade","INTEGER"]]},{"id": "3","operator": "Filter","condition": {"op": {"name": "=","kind": "EQUALS"},"operands": [{"input": 1},{"input": 1}]},"inputs": ["2"],"outputs": [["course_id","INTEGER"],["teacher_id","INTEGER"],["grade","INTEGER"]]},{"id": "4","operator": "HashAggregate","group": [1],"aggs": [{"agg": {"name": "COUNT","kind": "COUNT"},"type": {"type": "Int64","nullable": false},"distinct": false,"filter": -1,"approximate": false,"ignoreNulls": false,"operands": [],"name": null}],"mode": "None","inputs": ["3"],"outputs": [["teacher_id","INTEGER"],["$f1","BIGINT"]]},{"id": "5","operator": "Filter","condition": {"op": {"name": ">=","kind": "GREATER_THAN_OR_EQUAL"},"operands": [{"input": 1},{"literal": 2,"type": {"type": "Int64","nullable": false}}]},"inputs": ["4"],"outputs": [["teacher_id","INTEGER"],["$f1","BIGINT"]]},{"id": "6","operator": "HashJoin","condition": {"op": {"name": "=","kind": "EQUALS"},"operands": [{"input": 0},{"input": 2}]},"joinType": "semi","inputs": ["1","5"],"outputs": [["teacher_id","INTEGER"],["name","STRING"]]},{"id": "7","operator": "HashAggregate","group": [1],"aggs": [],"mode": "None","inputs": ["6"],"outputs": [["name","STRING"]]}]}"""
    #
    # # db_id可以不填，因为已经是relnode结构
    # db_id = ""
    #
    # prediction = model_eval(model=model,
    #                         device=device,
    #                         config=config,
    #                         g_sql_rel=g_rel_node,
    #                         p_sql_rel=p_rel_node,
    #                         db_id=db_id)
    #
    # print(prediction)
    get_GMN_score_df()



