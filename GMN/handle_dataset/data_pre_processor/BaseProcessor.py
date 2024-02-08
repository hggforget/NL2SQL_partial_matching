import numpy as np
import pandas as pd
from typing import Dict, List
import json
import hashlib


all_node_type = ['content_node', 'inputs', 'outputs', 'input', 'kind', 'operands', 'op', 'Project', 'exprs',
                 'TableScan', 'table', 'condition', 'literal', 'rels', 'joinType', 'HashJoin', 'Filter',
                 'HashAggregate', 'group', 'aggs', 'agg', 'field', 'direction', 'nulls', 'Sort', 'collation',
                 'Limit', 'fetch', 'distinct', 'NestedLoopJoin', 'all', 'Union', 'Minus', 'Window', 'window',
                 'agg_calls', 'partition_by', 'order_by', 'frame_type', 'boundary_type_start', 'boundary_type_end',
                 'Correlate', 'correlation', 'requiredColumns', 'Intersect', 'expr', 'correl', 'Values', 'tuples']
all_node_type_data = pd.Series(all_node_type)
computing_one_hot_encoded = pd.get_dummies(all_node_type_data)
handling_output_remain_number = 0


class BaseDataPreProcessor:
    all_node_type = all_node_type
    computing_one_hot_encoded = computing_one_hot_encoded
    handling_output_remain_number = handling_output_remain_number

    def fix_json(self, j):
        j = j.replace('","error":""}', "")
        j = j.replace('{"logicPlanJson":"', "")
        j = j.replace("\\n", "")
        j = j.replace("\\", "")
        j = j.replace("INSERT", '"INSERT"')
        j = j.replace("Optional[]", '"Optional[]"')
        j = j.replace("  ", "")

        return j

    def simplify_ast(self, d):
        d_list = d['rels']
        new_list_1 = []

        for item in d_list:
            new_list_1.append(self.delete_node(item))

        return {'rels': new_list_1}

    def delete_node(self, val):
        if isinstance(val, Dict):

            removed_key = ['extraDigest', 'name', 'mode', 'type', 'id']
            other_key = ['distinct', 'approximate', 'ignoreNulls', 'filter', 'nullable']
            for key in removed_key:
                val.pop(key, None)

            for key in other_key:
                if key in val.keys():
                    if val[key] == False or val[key] == -1:
                        val.pop(key, None)

            for k, v in val.items():
                if k == 'table':
                    del v[0]
                    del v[0]

                if k == 'outputs':
                    out = []
                    for o in v:
                        out.append(o[0])
                    val[k] = out

                self.delete_node(v)

        elif isinstance(val, List):
            for i in val:
                self.delete_node(i)

        return val

    def read_sql_rel(self, sql_rel, db_id):

        sql_rel = sql_rel.replace("___" + db_id, "")
        sql_rel = self.fix_json(sql_rel)
        try:
            sql_rel = json.loads(sql_rel)
        except Exception as e:
            print(e)
            sql_rel = {}

        return sql_rel

    def pack_batch(self, graphs):
        return NotImplemented

    def create_graph(self, rel, last, node_type, node_emb, node_value, edge, edge_type, mask_com, output_map):
        return NotImplemented

    def sql_to_graph(self, sql_rel, db_id):

        sql_rel = self.read_sql_rel(sql_rel, db_id)
        sql_rel_sim = self.simplify_ast(sql_rel)

        node_type, node_emb, node_value, edge, edge_type, mask_com, output_map = [], [], [], [], [], [], {}

        self.create_graph(sql_rel_sim, -1, node_type, node_emb, node_value, edge, edge_type, mask_com, output_map)
        sql_rel_sim_map = {"node_type": node_type, "node_emb": node_emb, "node_value": node_value, "edge": edge,
                           "edge_type": edge_type, "mask_com": mask_com, "output_map": output_map}

        return sql_rel_sim_map

    def string_to_hash_vector(self, input_str):
        h = hashlib.blake2b(key=b'bytedance key', digest_size=16)
        byte_string = input_str.encode('utf-8')
        h.update(byte_string)
        hash_hex = h.hexdigest()

        hash_bin = bin(int(hash_hex, 16))[2:].zfill(256)
        # 将二进制字符串转化为整数向量
        vector = [int(bit) for bit in hash_bin]
        return np.array(vector)

    def init_output_map(self, output_map, inputs, index):
        init_value = {"output_id": [], "output_name": [], "input": inputs, "operator_id": index, "include_id": [index]}
        if not output_map:
            output_map["0"] = init_value
        else:
            max_key = max(output_map, key=int)
            new_key = str(int(max_key) + 1)
            output_map[new_key] = init_value
        return output_map

    def isinstanceIntList(self, test_list):
        if isinstance(test_list, list):
            # 检查列表中的所有元素是否都是整数
            if all(isinstance(item, int) for item in test_list):
                return True
        return False