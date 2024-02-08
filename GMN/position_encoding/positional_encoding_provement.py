import typing

import torch
import time
import torch.nn.functional as F
import concurrent.futures

from GMN.handle_dataset.data_pre_processor.positional_encoding_processor import PositionalEncodingProcessor
from GMN.utils_file.predict_checkpoint import load_data_from_excel
from GMN.graphmatchingnetwork import PAIRWISE_SIMILARITY_FUNCTION
from GMN.evaluation import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout

def effective_rank(A, epsilon=1e-10):
    # 计算奇异值
    singular_values = np.linalg.svd(A, compute_uv=False)
    # 过滤掉小于阈值的奇异值
    singular_values = singular_values[singular_values > epsilon]
    # 归一化奇异值以得到概率分布
    normalized_singular_values = singular_values / np.sum(singular_values)
    # 计算熵
    entropy = -np.sum(normalized_singular_values * np.log(normalized_singular_values))
    # 计算并返回 effective rank
    return np.exp(entropy)


def effective_rank_threshold(X, threshold=0.001):
    """
    Calculate the effective rank of a matrix X.

    Parameters:
    X (numpy.ndarray): The data matrix.
    threshold (float): A threshold for determining significance of singular values.

    Returns:
    int: The effective rank of the matrix.
    """
    # Perform Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(X)

    # Normalize singular values
    normalized_singular_values = S / np.sum(S)

    # Count the number of significant singular values
    effective_rank = np.sum(normalized_singular_values > threshold)

    return effective_rank


def post_process(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        result = torch.sum(result).item()
        return result

    return wrapper


def get_diff(row):
    g_rel_node, p_rel_node, db_id = row.G_rel, row.P_rel, row.db_id
    data_processor = PositionalEncodingProcessor()
    pair_graph = data_processor.read_pair(g_rel_node, p_rel_node, db_id)[0]
    ground_truth, prediction, match_edge = pair_graph
    # same_subtrees = same_subtree_extractor(ground_truth, prediction)
    # return same_subtrees


    # 对比一下
    g_positional_encoding_whole, p_positional_encoding_whole = data_processor.pair_positional_encoding(
        pair_graph,
        abs_pe_dim=32)
    g_positional_encoding_separated, p_positional_encoding_separated = data_processor.pair_positional_encoding(
        pair_graph,
        abs_pe_dim=32,
        whole_graph=False
    )
    # g_table_scan_positional_encoding_whole = F.normalize(g_positional_encoding_whole[g_table_scan], dim=-1)
    # p_table_scan_positional_encoding_whole = F.normalize(p_positional_encoding_whole[p_table_scan], dim=-1)
    # g_table_scan_positional_encoding_separated = F.normalize(g_positional_encoding_separated[g_table_scan], dim=-1)
    # p_table_scan_positional_encoding_separated = F.normalize(p_positional_encoding_separated[p_table_scan], dim=-1)
    # g_table_scan_positional_encoding_whole = g_positional_encoding_whole[g_table_scan]
    # p_table_scan_positional_encoding_whole = p_positional_encoding_whole[p_table_scan]
    # g_table_scan_positional_encoding_separated = g_positional_encoding_separated[g_table_scan]
    # p_table_scan_positional_encoding_separated = p_positional_encoding_separated[p_table_scan]

    # whole_table_scan_similarity = torch.sum(g_table_scan_positional_encoding_whole -
    #                                         p_table_scan_positional_encoding_whole)
    # return torch.unsqueeze(whole_table_scan_similarity, dim=-1)
    # seperated_table_scan_similarity = torch.sum(g_table_scan_positional_encoding_separated -
    #                                             p_table_scan_positional_encoding_separated)
    # return torch.unsqueeze(whole_table_scan_similarity, dim=-1)


@post_process
def diff_func(x, y, func_name='euclidean'):
    val = PAIRWISE_SIMILARITY_FUNCTION[func_name](x, y)
    return val


def draw_pic(g_node_color, p_node_color, min_color, max_color):
    # 为图g和p的节点分配颜色
    # 我们将颜色值归一化到[0, 1]区间，以便于使用matplotlib的colormap
    g_color_values = (g_node_color - min_color) / (max_color - min_color)
    p_color_values = (p_node_color - min_color) / (max_color - min_color)
    g_color_values = g_color_values.tolist()
    p_color_values = p_color_values.tolist()
    # 创建两个图g和p
    G = nx.Graph()
    P = nx.Graph()

    # 添加节点到图g和p
    for i in range(1, 20):
        G.add_node(i, label=i, node_color=g_color_values[i-1])
    for i in range(1, 18):
        P.add_node(i, label=i, node_color=p_color_values[i-1])

    # 添加边到图g和p
    G.add_edges_from(edges_g)
    P.add_edges_from(edges_p)

    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # 绘制图g
    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos, node_color=[G.nodes[n]['node_color'] for n in G.nodes()], with_labels=False, cmap=plt.cm.viridis, ax=ax1)
    ax1.set_title('Graph G')

    # 绘制图p
    pos = graphviz_layout(P, prog="dot")
    nx.draw(P, pos, node_color=[P.nodes[n]['node_color'] for n in P.nodes()], with_labels=False, cmap=plt.cm.viridis, ax=ax2)
    ax2.set_title('Graph P')

    # 创建一个颜色条，共用于两个图
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min_color, vmax=max_color))
    sm._A = []  # 创建一个空数组用于ScalarMappable
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 添加颜色条的轴
    fig.colorbar(sm, cax=cbar_ax)  # 设置颜色条

    # 调整子图位置，为颜色条留出空间
    plt.subplots_adjust(right=0.9)

    plt.savefig("pe.png", dpi=2000)  # 将图保存为文件并指定dpi
    plt.show()


if __name__ == '__main__':
    edges_g = [
        [1, 2], [1, 3], [1, 4],
        [2, 5], [2, 6],
        [5, 7],
        [6, 8], [6, 19],
        [3, 9], [3, 10],
        [9, 11],
        [11, 12], [11, 13],
        [10, 14],
        [14, 15], [14, 16],
        [4, 17],
        [17, 18]
    ]
    edges_p = [
        [1, 2], [1, 3], [1, 4],
        [2, 5], [2, 6],
        [5, 7],
        [6, 8], [6, 12],
        [3, 9], [3, 10],
        [9, 11],
        [11, 13],
        [10, 14],
        [14, 15],
        [4, 16],
        [16, 17]
    ]

    ground_truth = {}
    ground_truth['edge_type'], \
    ground_truth['node_emb'],\
    ground_truth['edge'], \
    ground_truth['node_type'], \
    ground_truth['mask_com'], \
    ground_truth['node_value'] \
        = [np.array([1, 0, 0]) for _ in range(len(edges_g))], [np.array([_]) for _ in range(1, 20)] \
        , (torch.tensor(edges_g) - 1).tolist(), [_ for _ in range(1, 20)], [], []

    prediction = {}
    prediction['edge_type'], \
    prediction['node_emb'], \
    prediction['edge'], \
    prediction['node_type'], \
    prediction['mask_com'], \
    prediction['node_value'] \
        = [np.array([1, 0, 0]) for _ in range(len(edges_p))], [np.array([_]) for _ in range(1, 18)], \
        (torch.tensor(edges_p) - 1).tolist(), [_ for _ in range(1, 18)], [], []
    pair_graph = ground_truth, prediction, []

    # g_rel_node = """{"rels": [{"id": "0","operator": "TableScan","table": ["HIVE_BOE","spider","transcripts___student_transcripts_tracking"],"extraDigest": "StatsTable","inputs": [],"outputs": [["transcript_id","INTEGER"],["transcript_date","TIMESTAMP(9)"],["other_details","STRING"]]},{"id": "1","operator": "Project","outputs": [["transcript_date","TIMESTAMP(9)"]],"exprs": [{"input": 1}],"inputs": ["0"]},{"id": "2","operator": "Sort","collation": [{"field": 0,"direction": "DESCENDING","nulls": "LAST"}],"inputs": ["1"],"outputs": [["transcript_date","TIMESTAMP(9)"]]},{"id": "3","operator": "Limit","fetch": {"literal": 1,"type": {"type": "Int32","nullable": false}},"inputs": ["2"],"outputs": [["transcript_date","TIMESTAMP(9)"]]}]}"""
    # p_rel_node = """{"rels": [{"id": "0","operator": "TableScan","table": ["HIVE_BOE","spider","transcripts___student_transcripts_tracking"],"extraDigest": "StatsTable","inputs": [],"outputs": [["transcript_id","INTEGER"],["transcript_date","TIMESTAMP(9)"],["other_details","STRING"]]},{"id": "1","operator": "HashAggregate","group": [],"aggs": [{"agg": {"name": "MAX","kind": "MAX"},"type": {"type": "Nullable(DateTime)","nullable": true},"distinct": false,"filter": -1,"approximate": false,"ignoreNulls": false,"operands": [1],"name": "EXPR$0"}],"mode": "None","inputs": ["0"],"outputs": [["EXPR$0","TIMESTAMP(9)"]]}]}"""
    db_id = 'student_transcripts_tracking'
    g_rel_node = """{"rels": [{"id": "0","operator": "TableScan","table": ["HIVE_BOE","spider_cre_doc_template_mgt","paragraphs"],"extraDigest": "StatsTable","inputs": [],"outputs": [["paragraph_id","INTEGER"],["document_id","INTEGER"],["paragraph_text","STRING"],["other_details","STRING"]]},{"id": "1","operator": "HashAggregate","group": [1],"aggs": [{"agg": {"name": "COUNT","kind": "COUNT"},"type": {"type": "Int64","nullable": false},"distinct": false,"filter": -1,"approximate": false,"ignoreNulls": false,"operands": [],"name": null}],"mode": "None","inputs": ["0"],"outputs": [["document_id","INTEGER"],["$f1","BIGINT"]]},{"id": "2","operator": "Filter","condition": {"op": {"name": ">=","kind": "GREATER_THAN_OR_EQUAL"},"operands": [{"input": 1},{"literal": 2,"type": {"type": "Int64","nullable": false}}]},"inputs": ["1"],"outputs": [["document_id","INTEGER"],["$f1","BIGINT"]]},{"id": "3","operator": "Project","outputs": [["document_id","INTEGER"]],"exprs": [{"input": 0}],"inputs": ["2"]}]}"""
    p_rel_node = """{"rels": [{"id": "0","operator": "TableScan","table": ["HIVE_BOE","spider_cre_doc_template_mgt","paragraphs"],"extraDigest": "StatsTable","inputs": [],"outputs": [["paragraph_id","INTEGER"],["document_id","INTEGER"],["paragraph_text","STRING"],["other_details","STRING"]]},{"id": "1","operator": "HashAggregate","group": [1],"aggs": [{"agg": {"name": "COUNT","kind": "COUNT"},"type": {"type": "Int64","nullable": false},"distinct": false,"filter": -1,"approximate": false,"ignoreNulls": false,"operands": [],"name": null}],"mode": "None","inputs": ["0"],"outputs": [["document_id","INTEGER"],["$f1","BIGINT"]]},{"id": "2","operator": "Filter","condition": {"op": {"name": ">","kind": "GREATER_THAN"},"operands": [{"input": 1},{"literal": 2,"type": {"type": "Int64","nullable": false}}]},"inputs": ["1"],"outputs": [["document_id","INTEGER"],["$f1","BIGINT"]]},{"id": "3","operator": "Project","outputs": [["document_id","INTEGER"]],"exprs": [{"input": 0}],"inputs": ["2"]}]}"""

    # # 创建ThreadPoolExecutor，指定线程数量
    # # results = []
    data_processor = PositionalEncodingProcessor()
    # pair_graph = data_processor.read_pair(g_rel_node, p_rel_node, db_id)[0]
    # ground_truth, prediction, match_edge = pair_graph
    # same_subtrees = same_subtree_extractor(ground_truth, prediction)
    # return same_subtrees

    # 对比一下
    # g_positional_encoding_se, p_positional_encoding_se = data_processor.pair_positional_encoding(
    #     pair_graph,
    #     abs_pe_dim=16,
    #     whole_graph=False
    # )
    # g_node_color_se = torch.sum(g_positional_encoding_se, dim=-1).tolist()
    # p_node_color_se = torch.sum(p_positional_encoding_se, dim=-1).tolist()

    g_positional_encoding_wh, p_positional_encoding_wh = data_processor.pair_positional_encoding(
        pair_graph,
        abs_pe_dim=4
    )
    g_node_color_wh = torch.mean(g_positional_encoding_wh, dim=-1).tolist()
    p_node_color_wh = torch.mean(p_positional_encoding_wh, dim=-1).tolist()


    # # 定义节点颜色和边
    # g_node_color_se = np.array(g_node_color_se)  # 将列表转换为NumPy数组
    # p_node_color_se = np.array(p_node_color_se)  # 将列表转换为NumPy数组

    g_node_color_wh = np.array(g_node_color_wh)  # 将列表转换为NumPy数组
    p_node_color_wh = np.array(p_node_color_wh)  # 将列表转换为NumPy数组

    # 计算所有节点颜色的最小值和最大值
    all_colors = np.concatenate((# g_node_color_se, p_node_color_se,
                                  g_node_color_wh, p_node_color_wh
                                 ))
    min_color = np.min(all_colors)
    max_color = np.max(all_colors)
    # draw_pic(g_node_color_se, p_node_color_se, min_color, max_color)
    draw_pic(g_node_color_wh, p_node_color_wh, min_color, max_color)

