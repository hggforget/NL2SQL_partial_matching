from collections import Counter

import numpy
import pandas as pd
from tqdm import tqdm
import json
from typing import Dict, List
import numpy as np
import ast
import hashlib
import torch
import re
from GMN.handle_dataset.data_pre_processor.match_edge_processor import DataPreProcessor


all_node_type = ['all_columns', 'value', 'from', 'select', 'name', 'eq', 'Query', 'where', 'on', 'literal', 'inner join', 'count', 'groupby', 'join', 'dense_rank', 'orderby', 'and', 'sort', 'limit', 'gt', 'in', 'distinct', 'having', 'avg', 'max', 'sum', 'lt', 'gte', 'min', 'nin', 'select_distinct', 'like', 'intersect', 'cast', 'or', 'neq', 'real', 'left join', 'except', 'lte', 'union', 'exists', 'missing', 'over', 'row_number', 'between', 'partitionby', 'with', 'not', 'char', 'add', 'using', 'concat', 'integer', 'div', 'decimal', 'not_like', 'sub', 'union_all', 'case', 'then', 'when', 'mul', 'double', 'float', 'natural join', 'coalesce', 'lower', 'substr', 'varchar', 'round', 'cross join', 'null', 'date', 'int', 'instr', 'rank', 'length', 'abs', 'not_between', 'substring']
data = pd.Series(all_node_type)
computing_one_hot_encoded = pd.get_dummies(data)


class ASTProcessor(DataPreProcessor):
    all_node_type = all_node_type
    computing_one_hot_encoded = computing_one_hot_encoded

    def create_graph_ast(self, rel, last, node, node_feature, edge, edge_type, mask_com):

        if isinstance(rel, Dict):
            for k, v in rel.items():
                self.all_node_type.append(k)
                array = self.computing_one_hot_encoded[k]
                array = array.astype(int).T.values
                node_embedding = np.zeros(96)
                node_embedding[: array.shape[0]] = array
                node_feature.append(np.concatenate((node_embedding, self.string_to_hash_vector(k))))
                node.append(k)
                mask_com.append(1)


                if last != -1:
                    index = len(node) - 1
                    edge.append([last, index])
                    edge_type.append("ast_edge")

                index = len(node) - 1

                self.create_graph_ast(v, index, node, node_feature, edge, edge_type, mask_com)

        elif isinstance(rel, List) and len(rel) == 0:
            rel = str(rel)
            ascii_values = [ord(char) for char in rel]
            node_embedding = np.zeros(96)
            node_embedding[: np.array(ascii_values).shape[0]] = np.array(ascii_values)
            node_feature.append(np.concatenate((node_embedding, self.string_to_hash_vector(rel))))
            node.append(rel)
            mask_com.append(0)

            index = len(node) - 1
            edge.append([last, index])
            edge_type.append("ast_edge")


        elif isinstance(rel, List):
            for v in rel:
                self.create_graph_ast(v, last, node, node_feature, edge, edge_type, mask_com)


        elif type(rel) in [float, bool, int, str]:
            rel = str(rel)
            if len(rel) > 96:
                rel = rel[:96]

            ascii_values = [ord(char) for char in str(rel) if 0 <= ord(char) <= 127]
            node_embedding = np.zeros(96)
            node_embedding[: np.array(ascii_values).shape[0]] = np.array(ascii_values)
            node_feature.append(np.concatenate((node_embedding, self.string_to_hash_vector(rel))))
            node.append(rel)
            mask_com.append(0)

            index = len(node) - 1
            edge.append([last, index])
            edge_type.append("ast_edge")

    def replace_decimal(self, string):
        # 定义要搜索的模式 - 匹配 {'decimal': [任意数字, 任意数字]}
        pattern = r"\{'decimal': \[\d+, \d+\]\}"
        # 替换匹配的字符串为 "decimal"
        replaced_string = re.sub(pattern, "'decimal'", string)

        return replaced_string

    def fix_json(self, sql_rel):

        temp_sql_rel = sql_rel
        # sql_rel = sql_rel.replace("'", '"')
        # sql_rel = sql_rel.replace("True", '"True"')
        # sql_rel = sql_rel.replace("False", '"False"')
        # try:
        #     sql_rel = json.loads(sql_rel)
        # except:
        #     print(sql_rel)
        sql_rel = {"Query": eval(sql_rel)}

        return sql_rel

    def sql_to_graph_ast(self, sql_rel):

        sql_rel = self.fix_json(sql_rel)
        node, node_feature, edge, edge_type, mask_com = [], [], [], [], []
        self.create_graph_ast(sql_rel, -1, node, node_feature, edge, edge_type, mask_com)
        sql_rel_sim_map = {"node": node, "node_emb": node_feature, "edge": edge,
                           "edge_type": edge_type, "mask_com": mask_com}

        return sql_rel_sim_map

    def read_data(self, data):

        pairs, labels = [], []

        for i in tqdm(range(len(data))):
            p_sql_rel = data.loc[i]['P_ast']
            g_sql_rel = data.loc[i]['G_ast']
            label = data.loc[i]['label']

            if int(label) == 0:
                label = -1
            elif int(label) == 1:
                label = 1

            pairs.append((self.sql_to_graph_ast(g_sql_rel), self.sql_to_graph_ast(p_sql_rel)))
            labels.append(label)

        return pairs, labels

    @staticmethod
    def sort_elements_by_frequency(lst):
        # 使用Counter对象统计元素出现频率
        counts = Counter(lst)

        # 按出现频率对元素进行排序，出现频率多的在前
        # Counter.most_common()方法直接按计数多少返回一个排好序的元素列表
        sorted_elements_by_count = [element for element, count in counts.most_common()]

        return sorted_elements_by_count

    def pack_batch(self, graphs):
        from_idx = []
        to_idx = []
        graph_idx = []
        node_features = []
        edge_features = []
        mask_com = []
        n_total_nodes = 0
        n_total_edges = 0

        for i, pair_graph in enumerate(graphs):
            ground_truth = pair_graph[0]
            prediction = pair_graph[1]

            g_node_fea = ground_truth['node_emb']
            p_node_fea = prediction['node_emb']

            g_n_nodes = len(g_node_fea)
            p_n_nodes = len(p_node_fea)

            g_edges = torch.tensor(ground_truth['edge'], dtype=torch.int32)
            p_edges = torch.tensor(prediction['edge'], dtype=torch.int32)

            from_idx.append(g_edges[:, 0] + n_total_nodes)
            to_idx.append(g_edges[:, 1] + n_total_nodes)

            # edge_features += ground_truth['edge_type']
            n_total_nodes += g_n_nodes

            # edge_features += [np.array([0, 0, 1]) for i in range(len(match_edge))]

            from_idx.append(p_edges[:, 0] + n_total_nodes)
            to_idx.append(p_edges[:, 1] + n_total_nodes)
            n_total_nodes += p_n_nodes

            # edge_features += prediction['edge_type']

            g_idx = i * 2
            p_idx = (i * 2) + 1

            graph_idx.append(torch.ones(g_n_nodes, dtype=torch.int32) * g_idx)
            graph_idx.append(torch.ones(p_n_nodes, dtype=torch.int32) * p_idx)

            node_features += (g_node_fea + p_node_fea)

            mask_com.append(np.array(ground_truth['mask_com']))
            mask_com.append(np.array(prediction['mask_com']))

            n_total_edges += len(ground_truth['edge']) + len(prediction['edge'])

        edge_features = [np.array([1]) for i in range(n_total_edges)]
        from_idx = torch.cat(from_idx).long()
        to_idx = torch.cat(to_idx).long()
        edge_features = torch.from_numpy(np.array(edge_features, dtype=np.float32))
        edge_tuple = torch.cat((from_idx.unsqueeze(1), to_idx.unsqueeze(1), edge_features), dim=1)

        graph_idx = torch.cat(graph_idx).long()
        node_features = torch.from_numpy(np.array(node_features, dtype=np.float32))
        mask_com = torch.from_numpy(np.concatenate(mask_com))

        node_tuple = torch.cat((graph_idx.unsqueeze(1), mask_com.unsqueeze(1), node_features), dim=1)

        n_graphs = len(graphs) * 2
        return edge_tuple, node_tuple, n_graphs