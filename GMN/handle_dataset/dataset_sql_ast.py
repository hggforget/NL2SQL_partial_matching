from collections import Counter

import numpy
import pandas as pd
from tqdm import tqdm
import json
from typing import Dict, List
import numpy as np
import ast
import hashlib
import re




def string_to_hash_vector(input_str):
    h = hashlib.blake2b(key=b'bytedance key', digest_size=16)
    byte_string = input_str.encode('utf-8')
    h.update(byte_string)
    hash_hex = h.hexdigest()

    hash_bin = bin(int(hash_hex, 16))[2:].zfill(256)
    # 将二进制字符串转化为整数向量
    vector = [int(bit) for bit in hash_bin]
    return numpy.array(vector)


def create_graph(rel, last, node, node_feature, edge, edge_type, mask_com):

    global all_node_type

    if isinstance(rel, Dict):
        for k, v in rel.items():
            all_node_type.append(k)
            array = computing_one_hot_encoded[k]
            array = array.astype(int).T.values
            node_embedding = np.zeros(96)
            node_embedding[: array.shape[0]] = array
            node_feature.append(np.concatenate((node_embedding, string_to_hash_vector(k))))
            node.append(k)
            mask_com.append(1)


            if last != -1:
                index = len(node) - 1
                edge.append([last, index])
                edge_type.append("ast_edge")

            index = len(node) - 1

            create_graph(v, index, node, node_feature, edge, edge_type, mask_com)

    elif isinstance(rel, List) and len(rel) == 0:
        rel = str(rel)
        ascii_values = [ord(char) for char in rel]
        node_embedding = np.zeros(96)
        node_embedding[: np.array(ascii_values).shape[0]] = np.array(ascii_values)
        node_feature.append(np.concatenate((node_embedding, string_to_hash_vector(rel))))
        node.append(rel)
        mask_com.append(0)

        index = len(node) - 1
        edge.append([last, index])
        edge_type.append("ast_edge")


    elif isinstance(rel, List):
        for v in rel:
            create_graph(v, last, node, node_feature, edge, edge_type, mask_com)


    elif type(rel) in [float, bool, int, str]:
        rel = str(rel)
        if len(rel) > 96:
            rel = rel[:96]

        ascii_values = [ord(char) for char in str(rel) if 0 <= ord(char) <= 127]
        node_embedding = np.zeros(96)
        node_embedding[: np.array(ascii_values).shape[0]] = np.array(ascii_values)
        node_feature.append(np.concatenate((node_embedding, string_to_hash_vector(rel))))
        node.append(rel)
        mask_com.append(0)

        index = len(node) - 1
        edge.append([last, index])
        edge_type.append("ast_edge")


all_node_type = ['value', 'from', 'select', 'name', 'eq', 'Query', 'where', 'on', 'literal', 'inner join', 'count', 'groupby', 'join', 'orderby', 'and', 'sort', 'limit', 'gt', 'in', 'distinct', 'having', 'avg', 'max', 'sum', 'lt', 'gte', 'min', 'nin', 'select_distinct', 'like', 'intersect', 'cast', 'or', 'neq', 'real', 'left join', 'except', 'lte', 'union', 'exists', 'missing', 'over', 'row_number', 'between', 'partitionby', 'with', 'not', 'char', 'add', 'using', 'concat', 'integer', 'div', 'decimal', 'not_like', 'sub', 'union_all', 'case', 'then', 'when', 'mul', 'double', 'float', 'natural join', 'coalesce', 'lower', 'substr', 'varchar', 'round', 'cross join', 'null', 'date', 'int', 'instr', 'rank', 'length', 'abs', 'not_between', 'substring']
data = pd.Series(all_node_type)
computing_one_hot_encoded = pd.get_dummies(data)


def replace_decimal(string):
    # 定义要搜索的模式 - 匹配 {'decimal': [任意数字, 任意数字]}
    pattern = r"\{'decimal': \[\d+, \d+\]\}"
    # 替换匹配的字符串为 "decimal"
    replaced_string = re.sub(pattern, "'decimal'", string)

    return replaced_string

def fix_json(sql_rel):


    sql_rel = sql_rel.replace("'", '"')
    sql_rel = sql_rel.replace("True", '"True"')
    sql_rel = sql_rel.replace("False", '"False"')
    sql_rel = json.loads(sql_rel)
    sql_rel = {"Query": sql_rel}

    return sql_rel

def sql_to_graph(sql_rel):

    sql_rel = fix_json(sql_rel)
    node, node_feature, edge, edge_type, mask_com = [], [], [], [], []
    create_graph(sql_rel, -1, node, node_feature, edge, edge_type, mask_com)
    sql_rel_sim_map = {"node": node, "node_emb": node_feature, "edge": edge,
                         "edge_type": edge_type, "mask_com": mask_com}

    return sql_rel_sim_map


def read_data(data):

    pairs, labels = [], []

    for i in tqdm(range(len(data))):
        p_sql_rel = data.loc[i]['sql_ast_p']
        g_sql_rel = data.loc[i]['sql_ast_g']
        label = data.loc[i]['label']

        if int(label) < 3:
            label = -1
        elif int(label) >= 3:
            label = 1

        pairs.append((sql_to_graph(g_sql_rel), sql_to_graph(p_sql_rel)))
        labels.append(label)

    return pairs, labels

def sort_elements_by_frequency(lst):
    # 使用Counter对象统计元素出现频率
    counts = Counter(lst)

    # 按出现频率对元素进行排序，出现频率多的在前
    # Counter.most_common()方法直接按计数多少返回一个排好序的元素列表
    sorted_elements_by_count = [element for element, count in counts.most_common()]

    return sorted_elements_by_count

if __name__ == "__main__":

    all = []

    df1 = pd.read_excel('/Users/bytedance/Desktop/个人文件夹/pythonProject/NL2SQL_partial_matching/GMN/database_spider/dev.xlsx', "Sheet1")
    df2 = pd.read_excel( '/Users/bytedance/Desktop/个人文件夹/pythonProject/NL2SQL_partial_matching/GMN/database_spider/train.xlsx', "Sheet1")

    for i in range(len(df1)):

        p_sql_rel = df1.iloc[i]['sql_ast_p']
        g_sql_rel = df1.iloc[i]['sql_ast_g']

        p_graph = sql_to_graph(p_sql_rel)
        g_graph = sql_to_graph(g_sql_rel)

    all = all + all_node_type

    for i in range(len(df2)):
        p_sql_rel = df2.iloc[i]['sql_ast_p']
        g_sql_rel = df2.iloc[i]['sql_ast_g']

        p_graph = sql_to_graph(p_sql_rel)
        g_graph = sql_to_graph(g_sql_rel)


    all = all + all_node_type
    all = sort_elements_by_frequency(all)

    print(all)
    print(len(all))


