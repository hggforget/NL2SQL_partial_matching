import torch
import numpy as np
from tqdm import tqdm
import ast
from torch_geometric.data import Data
from GMN.position_encoding.data import GraphDataset
from GMN.handle_dataset.data_pre_processor.match_edge_processor import DataPreProcessor
from GMN.position_encoding.position_encoding import POSENCODINGS
from GMN.configure import get_default_config
from GMN.position_encoding.utils import same_subtree_extractor
config = get_default_config()


class PositionalEncodingProcessor(DataPreProcessor):

    @staticmethod
    def embedding_pair(pair_graph):
        ground_truth = pair_graph[0]
        prediction = pair_graph[1]
        match_edge = pair_graph[2]
        same_subtrees = same_subtree_extractor(ground_truth, prediction)
        g_node_fea = ground_truth['node_emb']
        p_node_fea = prediction['node_emb']

        g_n_nodes = len(g_node_fea)
        p_n_nodes = len(p_node_fea)

        g_edges = torch.tensor(ground_truth['edge'], dtype=torch.int32)
        p_edges = torch.tensor(prediction['edge'], dtype=torch.int32)
        g_edge_type = ground_truth['edge_type']
        p_edge_type = prediction['edge_type']
        g_edge_type = torch.from_numpy(np.stack(g_edge_type))
        p_edge_type = torch.from_numpy(np.stack(p_edge_type))
        m_edges = torch.tensor(match_edge, dtype=torch.int32)
        g_mask_com = np.array(ground_truth['mask_com'])
        p_mask_com = np.array(prediction['mask_com'])
        g_num_edges = len(ground_truth['edge'])
        p_num_edges = len(prediction['edge'])
        return g_node_fea, p_node_fea, g_n_nodes, p_n_nodes, \
            g_edges, p_edges, match_edge, g_mask_com, p_mask_com, \
            g_num_edges, p_num_edges, g_edge_type, p_edge_type, same_subtrees

    def pair_positional_encoding(self, pair_graph, abs_pe_dim=8, abs_pe='rw', whole_graph=True):
        g_node_fea, p_node_fea, g_n_nodes, p_n_nodes, \
            g_edges, p_edges, match_edge, g_mask_com, \
            p_mask_com, g_num_edges, p_num_edges, g_edge_type,\
            p_edge_type, same_subtrees = self.embedding_pair(pair_graph)
        if whole_graph:
            g_edges = self.select_ast_edge(g_edges, g_edge_type)
            p_edges = self.select_ast_edge(p_edges, p_edge_type)
            # # 使用`any`函数沿着行的方向（dim=0），这将返回一个布尔值的tensor，表示每一列是否包含至少一个0
            # g_columns_with_zero = (g_edges == 0).any(dim=1)
            #
            # # 找出包含0的列的索引
            # g_zero_columns_indices = g_columns_with_zero.nonzero(as_tuple=True)[0]
            # g_zero_columns = g_edges[g_zero_columns_indices, :]
            # g_edges = torch.cat((g_edges,
            #                      torch.vstack((g_zero_columns[:, 0] + g_n_nodes, g_zero_columns[:, 1])).T), dim=0)


            # # 使用`any`函数沿着行的方向（dim=0），这将返回一个布尔值的tensor，表示每一列是否包含至少一个0
            # p_columns_with_zero = (p_edges == 0).any(dim=1)
            #
            # # 找出包含0的列的索引
            # p_zero_columns_indices = p_columns_with_zero.nonzero(as_tuple=True)[0]
            # p_zero_columns = p_edges[p_zero_columns_indices, :]
            # p_edges = torch.cat((p_edges,
            #                      torch.vstack((p_zero_columns[:, 0] - g_n_nodes, p_zero_columns[:, 1])).T), dim=0)
            same_subtree_edge = torch.tensor(same_subtrees, dtype=torch.int32)
            from_idx_tensor = torch.concat((g_edges[:, 0], same_subtree_edge[:, 0], torch.tensor([0]),
                                            p_edges[:, 0] + g_n_nodes), dim=0).long()
            to_idx_tensor = torch.concat((g_edges[:, 1], same_subtree_edge[:, 1] + g_n_nodes, torch.tensor([g_n_nodes]),
                                          p_edges[:, 1] + g_n_nodes), dim=0).long()
        else:
            from_idx_tensor = torch.concat((g_edges[:, 0], p_edges[:, 0] + g_n_nodes), dim=0).long()
            to_idx_tensor = torch.concat((g_edges[:, 1], p_edges[:, 1] + g_n_nodes),
                                         dim=0).long()
            same_subtree_edge = []
        edges = torch.vstack((from_idx_tensor, to_idx_tensor))
        pair_graph_data = Data(x=torch.tensor(np.array(g_node_fea + p_node_fea), dtype=torch.float32),
                               edge_index=torch.cat((torch.flip(edges, dims=[0]), edges), dim=1),
                               edge_attr=torch.tensor([1 for i in range(2 * g_num_edges + 2 * p_num_edges +
                                                                        same_subtree_edge.shape[0])]))
        abs_pe_list = self.position_encoding(
            graph_dataset=GraphDataset([pair_graph_data], degree=True, k_hop=3, se='gnn', use_subgraph_edge_attr=True),
            abs_pe_dim=abs_pe_dim,
            abs_pe=abs_pe
        )
        g_positional_encoding = abs_pe_list[0][:g_n_nodes]
        p_positional_encoding = abs_pe_list[0][g_n_nodes:]
        return g_positional_encoding, p_positional_encoding

    @staticmethod
    def select_ast_edge(edges, edge_type):
        ast_edge_columns = (edge_type == torch.tensor([1, 0, 0])).all(dim=1)
        ast_edge_indices = ast_edge_columns.nonzero(as_tuple=True)[0]
        return edges[ast_edge_indices, :]

    def pack_batch(self, graphs):
        from_idx = []
        to_idx = []
        graph_idx = []
        node_features = []
        edge_features = []
        mask_com = []
        n_total_nodes = 0
        n_total_edges = 0
        pair_graph_data_list = []
        g_nodes_list = []
        for i, pair_graph in enumerate(graphs):
            g_node_fea, p_node_fea, g_n_nodes, p_n_nodes, \
                g_edges, p_edges, match_edge, g_mask_com, \
                p_mask_com, g_num_edges, p_num_edges, g_edge_type, \
                p_edge_type, same_subtrees = self.embedding_pair(pair_graph)
            g_from_idx = g_edges[:, 0] + n_total_nodes
            g_to_idx = g_edges[:, 1] + n_total_nodes
            from_idx.append(g_from_idx)
            to_idx.append(g_to_idx)
            g_nodes_list.append(g_n_nodes)
            # edge_features += ground_truth['edge_type']

            if len(match_edge) == 0:
                n_total_nodes += g_n_nodes
            else:
                # from_idx.append(m_edges[:, 0] + n_total_nodes)
                n_total_nodes += g_n_nodes
                # to_idx.append(m_edges[:, 1] + n_total_nodes)

            # edge_features += [np.array([0, 0, 1]) for i in range(len(match_edge))]
            p_from_idx = p_edges[:, 0] + n_total_nodes
            p_to_idx = p_edges[:, 1] + n_total_nodes
            from_idx.append(p_from_idx)
            to_idx.append(p_to_idx)
            n_total_nodes += p_n_nodes

            # edge_features += prediction['edge_type']

            g_idx = i * 2
            p_idx = (i * 2) + 1

            graph_idx.append(torch.ones(g_n_nodes, dtype=torch.int32) * g_idx)
            graph_idx.append(torch.ones(p_n_nodes, dtype=torch.int32) * p_idx)

            node_features += (g_node_fea + p_node_fea)

            mask_com.append(g_mask_com)
            mask_com.append(p_mask_com)

            n_total_edges += g_num_edges + p_num_edges
            if config['abs_pe_embedding']:
                if config['pe_type'] == 'whole':
                    from_idx_tensor = torch.concat((g_edges[:, 0], torch.tensor([0]),
                                                    p_edges[:, 0] + g_n_nodes), dim=0).long()
                    to_idx_tensor = torch.concat((g_edges[:, 1], torch.tensor([g_n_nodes]),
                                                  p_edges[:, 1] + g_n_nodes), dim=0).long()
                    if same_subtrees:
                        same_subtree_edge = torch.tensor(same_subtrees, dtype=torch.int32)
                        from_idx_tensor = torch.concat((from_idx_tensor, same_subtree_edge[:, 0]), dim=0).long()
                        to_idx_tensor = torch.concat((to_idx_tensor, same_subtree_edge[:, 1] + g_n_nodes), dim=0).long()
                    edges = torch.vstack((from_idx_tensor, to_idx_tensor))
                    pair_graph_data = Data(x=torch.tensor(np.array(g_node_fea + p_node_fea), dtype=torch.float32),
                                           edge_index=torch.cat((torch.flip(edges, dims=[0]), edges), dim=1))
                    pair_graph_data_list.append(pair_graph_data)
                elif config['pe_type'] == 'separate':
                    from_idx_tensor = torch.concat((g_edges[:, 0], p_edges[:, 0] + g_n_nodes), dim=0).long()
                    to_idx_tensor = torch.concat((g_edges[:, 1], p_edges[:, 1] + g_n_nodes),
                                                 dim=0).long()
                    edges = torch.vstack((from_idx_tensor, to_idx_tensor))
                    pair_graph_data = Data(x=torch.tensor(np.array(g_node_fea + p_node_fea), dtype=torch.float32),
                                           edge_index=torch.cat((torch.flip(edges, dims=[0]), edges), dim=1))
                    pair_graph_data_list.append(pair_graph_data)
                else:
                    raise Exception(f'Unsupported PE type')

        edge_features = [np.array([1]) for i in range(n_total_edges)]
        from_idx = torch.cat(from_idx).long()
        to_idx = torch.cat(to_idx).long()
        edge_features = torch.from_numpy(np.array(edge_features, dtype=np.float32))
        edge_tuple = torch.cat((from_idx.unsqueeze(1), to_idx.unsqueeze(1), edge_features), dim=1)

        graph_idx = torch.cat(graph_idx).long()
        node_features = torch.from_numpy(np.array(node_features, dtype=np.float32))
        mask_com = torch.from_numpy(np.concatenate(mask_com))
        if config['abs_pe_embedding']:
            abs_pe_list = self.position_encoding(GraphDataset(
                pair_graph_data_list, degree=False, k_hop=3, se='gnn', use_subgraph_edge_attr=False))
            node_tuple = torch.cat((graph_idx.unsqueeze(1), mask_com.unsqueeze(1),
                                    node_features, torch.vstack(abs_pe_list),), dim=1)
        else:
            node_tuple = torch.cat((graph_idx.unsqueeze(1), mask_com.unsqueeze(1), node_features, ), dim=1)

        n_graphs = len(graphs) * 2

        return edge_tuple, node_tuple, n_graphs

    @staticmethod
    def position_encoding(graph_dataset, abs_pe_dim=config['abs_pe_dim'], abs_pe=config['abs_pe']):
        abs_pe_encoder = None
        if abs_pe and abs_pe_dim > 0:
            abs_pe_method = POSENCODINGS[abs_pe]
            abs_pe_encoder = abs_pe_method(abs_pe_dim, normalization='sym')
            if abs_pe_encoder is not None:
                abs_pe_encoder.apply_to(graph_dataset)
                return graph_dataset.abs_pe_list
