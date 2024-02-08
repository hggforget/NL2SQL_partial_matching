import random

from tqdm import tqdm
import numpy as np
import ast
from typing import Dict, List
import torch
from GMN.handle_dataset.data_pre_processor.BaseProcessor import BaseDataPreProcessor
from torch_geometric.data import Data
from GMN.position_encoding.data import GraphDataset


class DataPreProcessor(BaseDataPreProcessor):

    #
    # def __init__(self, g_sql_rel: str, p_sql_rel: str, db_id: str):
    #     ground_truth = self.sql_to_graph(g_sql_rel, db_id)
    #     prediction = self.sql_to_graph(p_sql_rel, db_id)
    #     match_edge = self.add_match_edge(ground_truth, prediction)
    #     self.pairs = [(ground_truth, prediction, match_edge)]

    def read_data(self, data):
        pairs, labels = [], []

        for i in tqdm(range(len(data))):
            try:
                p_sql_rel = data.loc[i]['P_rel']
                g_sql_rel = data.loc[i]['G_rel']
            except:
                print(i)

            try:
                label = data.loc[i]['label']
            except:
                label = data.loc[i]['label_new']
            db_id = data.loc[i]['db_id']
            if "[" and "]" in db_id:
                db_id = ast.literal_eval(db_id)
                db_id = db_id[0]

            if int(label) == 0:
                label = -1
            elif int(label) == 1:
                label = 1

            ground_truth = self.sql_to_graph(g_sql_rel, db_id)
            prediction = self.sql_to_graph(p_sql_rel, db_id)
            match_edge = self.add_match_edge(ground_truth, prediction)
            pairs.append((ground_truth, prediction, match_edge))
            labels.append(label)
        return pairs, labels

    def add_match_edge(self, ground_truth, prediction):
        match_edge = []
        ground_node_value = ground_truth['node_value']
        prediction_node_value = prediction['node_value']

        for g_i, g_value in enumerate(ground_node_value):
            if g_value is None or g_value not in prediction_node_value:
                continue

            for i, value in enumerate(prediction_node_value):

                if value == g_value:
                    match_edge.append([g_i, i])

        return match_edge

    def create_graph(self, rel, last, node_type, node_emb, node_value, edge, edge_type, mask_com, output_map):
        # 如果ok，当前节点不要append
        # {0 : {output : [2,3,4,6,7,8,3], input: []},
        #  1 : {output : [2,3,4,6,7,8,3], input: [0],
        #  2 : {output : [2,3,4,6,7,8,3], input: [0, 1]}}
        # Id, input
        #  input: [0, 1] -> [2,3,4,6,7,8,3] + [2,3,4,6,7,8,3] 的第七个


        if isinstance(rel, Dict):
            for k, v in rel.items():

                # array = computing_one_hot_encoded[k]
                # array = array.astype(int).T.values
                # node_emb.append(array)

                array = self.computing_one_hot_encoded[k]
                array = array.astype(int).T.values
                node_embedding = np.zeros(64)
                node_embedding[: array.shape[0]] = array
                node_emb.append(np.concatenate((node_embedding, self.string_to_hash_vector(k))))

                node_type.append(k)
                mask_com.append(1)
                node_value.append(None)

                if k != "rels":
                    max_key = max(output_map, key=int)
                    include_id = output_map[max_key]["include_id"]
                    include_id.append(len(node_type) - 1)
                    output_map[max_key]["include_id"] = include_id

                if last != -1:
                    index = len(node_type) - 1
                    edge.append([last, index])
                    # edge_type.append("ast_edge")
                    edge_type.append(np.array([1, 0, 0]))

                index = len(node_type) - 1
                # 如果碰到output，进入output handling 模式，需要继续深入
                # 例如：'outputs': ['name', 'score']

                if k == "outputs":
                    self.handling_output_remain_number = len(v)
                    # print(" handling_output_remain_number", handling_output_remain_number)
                # 如果碰到input，不要继续深入（达到删除子节点的效果），将当前的index指向
                # 例如：'outputs': ['name', 'score']
                if k in ["input", "group", "operands", "partition_by", "requiredColumns", "field"]:
                    # 找到output_map中最大的max_key
                    max_output_key = str(max(map(int, output_map.keys())))
                    # 获取最大output键的input值
                    max_output_input = output_map[max_output_key]['input']
                    # 根据input值拼接对应的 1/2/n个 列表
                    merge_outputs_list = []
                    for input_value in max_output_input:
                        merge_outputs_list.extend(output_map[str(input_value)]['output_id'])
                    # if k != 'input' and k != 'group':
                    #     print("v", v)
                    # input / group -> 单个int值
                    # operands -> 单个int 列表 或 字典列表 [{'input': 2}, {'literal': 0}]}]
                    if isinstance(v, int):
                        edge.append([index, merge_outputs_list[v]])
                        # edge_type.append("data_edge")
                        edge_type.append(np.array([0, 1, 0]))

                    elif self.isinstanceIntList(v):
                        for v_index in v:
                            edge.append([index, merge_outputs_list[v_index]])
                            # edge_type.append("data_edge")
                            edge_type.append(np.array([0, 1, 0]))
                    else:
                        self.create_graph(v, index, node_type, node_emb, node_value, edge, edge_type, mask_com, output_map)
                else:
                   self.create_graph(v, index, node_type, node_emb, node_value, edge, edge_type, mask_com, output_map)

        elif isinstance(rel, List) and len(rel) == 0:

            # array = computing_one_hot_encoded['content_node']
            # array = array.astype(int).T.values
            # node_emb.append(array)

            rel = str(rel)
            ascii_values = [ord(char) for char in rel]
            node_embedding = np.zeros(64)
            node_embedding[: np.array(ascii_values).shape[0]] = np.array(ascii_values)
            node_emb.append(np.concatenate((node_embedding, self.string_to_hash_vector(rel))))

            node_type.append('[]')
            mask_com.append(0)
            node_value.append('[]')

            index = len(node_type) - 1
            edge.append([last, index])
            # edge_type.append("ast_edge")
            edge_type.append(np.array([1, 0, 0]))

            max_key = max(output_map, key=int)
            include_id = output_map[max_key]["include_id"]
            include_id.append(index)
            output_map[max_key]["include_id"] = include_id


        elif isinstance(rel, List):
            for v in rel:

                if 'operator' in v:
                    # 每次碰到 'operator'，增加output_map的Key
                    # array = computing_one_hot_encoded[v['operator']]
                    # array = array.astype(int).T.values
                    # node_emb.append(array)

                    array = self.computing_one_hot_encoded[v['operator']]
                    array = array.astype(int).T.values
                    node_embedding = np.zeros(64)
                    node_embedding[: array.shape[0]] = array
                    node_emb.append(np.concatenate((node_embedding, self.string_to_hash_vector(v['operator']))))

                    node_type.append(v['operator'])
                    mask_com.append(1)
                    node_value.append(None)

                    edge.append([last, len(node_type) - 1])
                    # edge_type.append("ast_edge")
                    edge_type.append(np.array([1, 0, 0]))

                    del v['operator']
                    index = len(node_type) - 1
                    self.init_output_map(output_map, v['inputs'], index)
                    for input_id in v['inputs']:
                        logic_node = output_map[input_id]['operator_id']
                        edge.append([len(node_type) - 1, logic_node])
                        # edge_type.append("logic_edge")
                        edge_type.append(np.array([0, 1, 0]))

                    del v['inputs']
                    self.create_graph(v, index, node_type, node_emb, node_value, edge, edge_type, mask_com, output_map)

                else:
                    self.create_graph(v, last, node_type, node_emb, node_value, edge, edge_type, mask_com, output_map)

        elif type(rel) in [float, bool, int, str]:
            rel = str(rel)

            # array = computing_one_hot_encoded['content_node']
            # array = array.astype(int).T.values
            # node_emb.append(array)

            rel = str(rel)
            if len(rel) > 64:
                rel = rel[:64]

            ascii_values = [ord(char) for char in str(rel) if 0 <= ord(char) <= 127]
            node_embedding = np.zeros(64)
            node_embedding[: np.array(ascii_values).shape[0]] = np.array(ascii_values)
            node_emb.append(np.concatenate((node_embedding, self.string_to_hash_vector(rel))))

            node_type.append(rel)
            mask_com.append(0)
            node_value.append(rel)

            index = len(node_type) - 1
            edge.append([last, index])
            # edge_type.append("ast_edge")
            edge_type.append(np.array([1, 0, 0]))

            max_key = max(output_map, key=int)
            include_id = output_map[max_key]["include_id"]
            include_id.append(index)
            output_map[max_key]["include_id"] = include_id

            if self.handling_output_remain_number != 0:
                # 找到output_map中最大的max_key
                max_output_key = str(max(map(int, output_map.keys())))
                # 在最大的output键对应的output列表中添加元素1
                output_map[max_output_key]['output_id'].append(index)
                output_map[max_output_key]['output_name'].append(rel)
                self.handling_output_remain_number = self.handling_output_remain_number - 1

                input_output_id = []
                intput_output_name = []
                for input_id in output_map[max_output_key]['input']:
                    input_output_id += output_map[input_id]['output_id']
                    intput_output_name += output_map[input_id]['output_name']

                indexes = [index for index, value in enumerate(intput_output_name) if value == rel]
                for i in indexes:
                    edge.append([index, input_output_id[i]])
                    # edge_type.append("new_data_edge")
                    edge_type.append(np.array([0, 1, 0]))

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
            match_edge = pair_graph[2]

            g_node_fea = ground_truth['node_emb']
            p_node_fea = prediction['node_emb']

            g_n_nodes = len(g_node_fea)
            p_n_nodes = len(p_node_fea)

            g_edges = torch.tensor(ground_truth['edge'], dtype=torch.int32)
            p_edges = torch.tensor(prediction['edge'], dtype=torch.int32)
            m_edges = torch.tensor(match_edge, dtype=torch.int32)

            from_idx.append(g_edges[:, 0] + n_total_nodes)
            to_idx.append(g_edges[:, 1] + n_total_nodes)

            # edge_features += ground_truth['edge_type']

            if len(match_edge) == 0:
                n_total_nodes += g_n_nodes
            else:
                # from_idx.append(m_edges[:, 0] + n_total_nodes)
                n_total_nodes += g_n_nodes
                # to_idx.append(m_edges[:, 1] + n_total_nodes)

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

    def shuffle(self, pair_list, labels):
        data_with_label = [(pair, labels[idx]) for idx, pair in enumerate(pair_list)]
        random.shuffle(data_with_label)
        pair_list = [pair[0] for pair in data_with_label]
        labels = [pair[1] for pair in data_with_label]
        return pair_list, labels

    def pairs_spider(self, batch_size, pair_list, labels):
        pair_list, labels = self.shuffle(pair_list, labels)
        batch_data_list = []
        ptr = 0
        while ptr < len(pair_list):
            if ptr + batch_size > len(pair_list):
                next_ptr = len(pair_list)
            else:
                next_ptr = ptr + batch_size
            batch_graphs = pair_list[ptr: next_ptr]
            edge_tuple, node_tuple, n_graphs = self.pack_batch(batch_graphs)

            batch_data_list.append(
                [edge_tuple, node_tuple, n_graphs, torch.tensor(labels[ptr: ptr + batch_size], dtype=torch.int32)])
            ptr = next_ptr

        return batch_data_list

    def read_pair(self, g_sql_rel: str, p_sql_rel: str, db_id: str):

        ground_truth = self.sql_to_graph(g_sql_rel, db_id)
        prediction = self.sql_to_graph(p_sql_rel, db_id)
        match_edge = self.add_match_edge(ground_truth, prediction)
        return [(ground_truth, prediction, match_edge)]
