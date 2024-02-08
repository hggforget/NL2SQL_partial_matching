# -*- coding: utf-8 -*-
from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr
from torch_geometric.utils.num_nodes import maybe_num_nodes
from scipy.sparse import csr_matrix,lil_matrix
import numpy as np
import typing


def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def dense_to_sparse_tensor(matrix):
    rows, columns = torch.where(matrix > 0)
    values = torch.ones(rows.shape)
    indices = torch.from_numpy(np.vstack((rows,
                                          columns))).long()
    shape = torch.Size(matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def extract_node_feature(data, reduce='add'):
    if reduce in ['mean', 'max', 'add']:
        data.x = scatter(data.edge_attr,
                         data.edge_index[0],
                         dim=0,
                         dim_size=data.num_nodes,
                         reduce=reduce)
    else:
        raise Exception('Unknown Aggregation Type')
    return data


def pad_batch(x, ptr, return_mask=False):
    bsz = len(ptr) - 1
    # num_nodes = torch.diff(ptr)
    max_num_nodes = torch.diff(ptr).max().item()

    all_num_nodes = ptr[-1].item()
    cls_tokens = False
    x_size = len(x[0]) if isinstance(x, (list, tuple)) else len(x)
    if x_size > all_num_nodes:
        cls_tokens = True
        max_num_nodes += 1
    if isinstance(x, (list, tuple)):
        new_x = [xi.new_zeros(bsz, max_num_nodes, xi.shape[-1]) for xi in x]
        if return_mask:
            padding_mask = x[0].new_zeros(bsz, max_num_nodes).bool()
    else:
        new_x = x.new_zeros(bsz, max_num_nodes, x.shape[-1])
        if return_mask:
            padding_mask = x.new_zeros(bsz, max_num_nodes).bool()

    for i in range(bsz):
        num_node = ptr[i + 1] - ptr[i]
        if isinstance(x, (list, tuple)):
            for j in range(len(x)):
                new_x[j][i][:num_node] = x[j][ptr[i]:ptr[i + 1]]
                if cls_tokens:
                    new_x[j][i][-1] = x[j][all_num_nodes + i]
        else:
            new_x[i][:num_node] = x[ptr[i]:ptr[i + 1]]
            if cls_tokens:
                new_x[i][-1] = x[all_num_nodes + i]
        if return_mask:
            padding_mask[i][num_node:] = True
            if cls_tokens:
                padding_mask[i][-1] = False
    if return_mask:
        return new_x, padding_mask
    return new_x

def unpad_batch(x, ptr):
    bsz, n, d = x.shape
    max_num_nodes = torch.diff(ptr).max().item()
    num_nodes = ptr[-1].item()
    all_num_nodes = num_nodes
    cls_tokens = False
    if n > max_num_nodes:
        cls_tokens = True
        all_num_nodes += bsz
    new_x = x.new_zeros(all_num_nodes, d)
    for i in range(bsz):
        new_x[ptr[i]:ptr[i + 1]] = x[i][:ptr[i + 1] - ptr[i]]
        if cls_tokens:
            new_x[num_nodes + i] = x[i][-1]
    return new_x


class Node:

    def __init__(self, node_idx, nodel_emb, children, parent=None):
        self.node_idx = node_idx
        self.node_emb = nodel_emb
        self.children = children
        self.parent = parent

    def visit(self):
        nodes = [self.node_idx]
        for child in self.children:
            nodes.extend(child.visit())
        return nodes


class AST:

    def __init__(self, node_label, edges: typing.List[typing.List[int]]):
        self.node_label = node_label
        self.edge = [[] for _ in range(len(node_label))]
        self.nodes = []
        for edge in edges:
            from_idx, to_idx = edge
            self.edge[from_idx].append(to_idx)
        self.ast = self.construct_ast(0, self.edge[0])

    def construct_ast(self, root_idx, children, parent=None) -> Node:

        node = Node(root_idx, self.node_label[root_idx], [], parent)
        self.nodes.append(node)
        for child_idx in children:
            node.children.append(self.construct_ast(child_idx, self.edge[child_idx], node))
        return node

    @classmethod
    def compare_ast(cls, root1, root2) -> bool:

        if len(root1.children) != len(root2.children):
            return False
        if root1.node_emb != root2.node_emb:
            return False
        mask = []
        for child1 in root1.children:
            flag = False
            for child2 in root2.children:
                if child2 in mask:
                    continue
                if cls.compare_ast(child1, child2):
                    mask.append(child2)
                    flag = True
                    break
            if not flag:
                return False
        if len(mask) == len(root2.children):
            return True
        return False


def select_ast_edge(edges, edge_type):
    ast_edge_columns = (edge_type == torch.tensor([1, 0, 0])).all(dim=1)
    ast_edge_indices = ast_edge_columns.nonzero(as_tuple=True)[0]
    return edges[ast_edge_indices, :].tolist()


def same_subtree_extractor(ground_truth, prediction):
    g_edge = ground_truth['edge']
    p_edge = prediction['edge']
    g_node_emb = ground_truth['node_type']
    p_node_emb = prediction['node_type']
    g_edge_type = ground_truth['edge_type']
    p_edge_type = prediction['edge_type']
    g_edge_type = np.stack(g_edge_type)
    p_edge_type = np.stack(p_edge_type)
    g_edge = select_ast_edge(torch.tensor(g_edge, dtype=torch.int32), torch.from_numpy(g_edge_type))
    p_edge = select_ast_edge(torch.tensor(p_edge, dtype=torch.int32), torch.from_numpy(p_edge_type))

    g_ast = AST(g_node_emb, g_edge)
    p_ast = AST(p_node_emb, p_edge)
    g_subtrees_idx = g_ast.ast.children
    p_subtrees_idx = p_ast.ast.children
    res = []
    for g_subtree_idx in g_subtrees_idx:
        for p_subtree_idx in p_subtrees_idx:
            if AST.compare_ast(g_ast.nodes[g_subtree_idx.node_idx], p_ast.nodes[p_subtree_idx.node_idx]):
                res.append([g_subtree_idx.node_idx, p_subtree_idx.node_idx])
    return res
    # g_mapping = {subtree_tuple[0]: 0 for subtree_tuple in res}
    # p_mapping = {subtree_tuple[1]: 0 for subtree_tuple in res}
    # for subtree_tuple in res:
    #     g_mapping[subtree_tuple[0]] += 1
    #     p_mapping[subtree_tuple[1]] += 1
    # final_res = []
    # for subtree_tuple in res:
    #     if g_mapping[subtree_tuple[0]] == 1 and p_mapping[subtree_tuple[1]] == 1:
    #         final_res.append(subtree_tuple)
    # return final_res
    # g_table_scan_idx = g_ast.node_label.index('TableScan')
    # p_table_scan_idx = p_ast.node_label.index('TableScan')
    # g_table_scan = g_ast.nodes[g_table_scan_idx]
    # p_table_scan = p_ast.nodes[p_table_scan_idx]
    # if AST.compare_ast(g_table_scan, p_table_scan):
    #     return g_table_scan.visit(), p_table_scan.visit()
