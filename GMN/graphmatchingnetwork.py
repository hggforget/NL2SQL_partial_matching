import math

from GMN.graphembeddingnetwork import GraphEmbeddingNet
from GMN.graphembeddingnetwork import GraphPropLayer
from torch_geometric.data import Data
from GMN.position_encoding.data import GraphDataset
from GMN.position_encoding.position_encoding import RWEncoding
import torch
from torch import nn
import torch.nn.functional as F


def pairwise_euclidean_similarity(x, y):
    """Compute the pairwise Euclidean similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = -|x_i - y_j|^2.

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise euclidean similarity.
    """
    s = 2 * torch.mm(x, torch.transpose(y, 1, 0))
    diag_x = torch.sum(x * x, dim=-1)
    diag_x = torch.unsqueeze(diag_x, 1)
    diag_y = torch.reshape(torch.sum(y * y, dim=-1), (1, -1))

    return s - diag_x - diag_y


def pairwise_dot_product_similarity(x, y):
    """Compute the dot product similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j.

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise dot product similarity.
    """
    return torch.mm(x, torch.transpose(y, 1, 0))


def pairwise_cosine_similarity(x, y):
    """Compute the cosine similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j / (|x_i||y_j|).

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise cosine similarity.
    """
    return torch.mm(F.normalize(x), F.normalize(y).t())


PAIRWISE_SIMILARITY_FUNCTION = {
    'euclidean': pairwise_euclidean_similarity,
    'dotproduct': pairwise_dot_product_similarity,
    'cosine': pairwise_cosine_similarity,
}


def get_pairwise_similarity(name):
    """Get pairwise similarity metric by name.

    Args:
      name: string, name of the similarity metric, one of {dot-product, cosine,
        euclidean}.

    Returns:
      similarity: a (x, y) -> sim function.

    Raises:
      ValueError: if name is not supported.
    """
    # if name not in PAIRWISE_SIMILARITY_FUNCTION:
    #     raise ValueError('Similarity metric name "%s" not supported.' % name)
    # else:
    return PAIRWISE_SIMILARITY_FUNCTION[name]


def compute_cross_attention(x, y, sim, dim):
    """Compute cross attention.

    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i

    Args:
      x: NxD float tensor.
      y: MxD float tensor.
      sim: a (x, y) -> similarity function.

    Returns:
      attention_x: NxD float tensor.
      attention_y: NxD float tensor.
    """
    a = sim(x, y) / math.sqrt(dim)
    a_x = torch.softmax(a, dim=1)  # i->j
    a_y = torch.softmax(a, dim=0)  # j->i
    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y


def masked_softmax(similarity, mask):
    exps = torch.exp(similarity)
    masked_exps = exps * mask
    masked_sums = masked_exps.sum(2, keepdim=True) + 1e-10  # 防止除0
    return masked_exps / masked_sums


class GraphPropMatchingLayer(GraphPropLayer):
    """A graph propagation layer that also does cross graph matching.

    It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
    forms the first pair and graph 2 and 3 are the second pair etc., and computes
    cross-graph attention-based matching for each pair.
    """

    def __init__(self,
                 node_state_dim,
                 edge_state_dim,
                 abs_pe_dim,
                 edge_hidden_sizes,  # int
                 node_hidden_sizes,  # int
                 edge_net_init_scale=0.1,
                 node_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 prop_type='embedding',
                 name='graph-net',
                 abs_pe_embedding=False,
                 ):
        super().__init__(node_state_dim=node_state_dim,
                         edge_state_dim=edge_state_dim,
                         edge_hidden_sizes=edge_hidden_sizes,
                         node_hidden_sizes=node_hidden_sizes,
                         edge_net_init_scale=edge_net_init_scale,
                         node_update_type=node_update_type,
                         use_reverse_direction=use_reverse_direction,
                         reverse_dir_param_different=reverse_dir_param_different,
                         layer_norm=layer_norm,
                         prop_type=prop_type,
                         name=name)

        if abs_pe_embedding:
            self._abs_pe_dim = abs_pe_dim
            self._abs_pe_embedding = nn.Linear(node_state_dim, abs_pe_dim)

    def forward(self,
                node_states,
                from_idx,
                to_idx,
                edge_features=None,
                node_features=None,
                **args
                ):
        """Run one propagation step with cross-graph matching.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, node states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          graph_idx: [n_onodes] int tensor, graph id for each node.
          n_graphs: integer, number of graphs in the batch.
          similarity: type of similarity to use for the cross graph attention.
          edge_features: if not None, should be [n_edges, edge_feat_dim] tensor,
            extra edge features.
          node_features: if not None, should be [n_nodes, node_feat_dim] tensor,
            extra node features.

        Returns:
          node_states: [n_nodes, node_state_dim] float tensor, new node states.

        Raises:
          ValueError: if some options are not provided correctly.
        """
        graph_idx = args['graph_idx']
        n_graphs = args['n_graphs']
        similarity = args['similarity']
        node_abe_features = args['node_abe_features']
        aggregated_messages = self._compute_aggregated_messages(
            node_states, from_idx, to_idx, edge_features=edge_features)
        cross_graph_attention = self.batch_block_pair_attention(
            node_states, graph_idx, n_graphs, similarity=similarity,
            node_abe_features=node_abe_features)
        attention_input = node_states - cross_graph_attention
        return self._compute_node_update(node_states,
                                         [aggregated_messages, attention_input],
                                         node_features=node_features)

    def batch_block_pair_attention(self,
                                   data,
                                   block_idx,
                                   n_blocks,
                                   similarity='dotproduct',
                                   node_abe_features=None
                                   ):
        """Compute batched attention between pairs of blocks.

        This function partitions the batch data into blocks according to block_idx.
        For each pair of blocks, x = data[block_idx == 2i], and
        y = data[block_idx == 2i+1], we compute

        x_i attend to y_j:
        a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
        y_j attend to x_i:
        a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))

        and

        attention_x = sum_j a_{i->j} y_j
        attention_y = sum_i a_{j->i} x_i.

        Args:
          data: NxD float tensor.
          block_idx: N-dim int tensor.
          n_blocks: integer.
          similarity: a string, the similarity metric.

        Returns:
          attention_output: NxD float tensor, each x_i replaced by attention_x_i.

        Raises:
          ValueError: if n_blocks is not an integer or not a multiple of 2.
        """

        sim = get_pairwise_similarity(similarity)
        # results2 = []
        # # This is probably better than doing boolean_mask for each i
        # partitions2 = []
        # for i in range(n_blocks):
        #     partitions2.append(data[block_idx == i, :])
        #
        # for i in range(0, n_blocks, 2):
        #     x = partitions2[i]
        #     y = partitions2[i + 1]
        #     attention_x2, attention_y2 = compute_cross_attention(x, y, sim)
        #     results2.append(attention_x2)
        #     results2.append(attention_y2)
        # results2 = torch.cat(results2, dim=0)
        # return results2

        results = []
        # Optimized partitioning
        sorted_indices = torch.argsort(block_idx)
        sorted_data = data[sorted_indices]
        sorted_node_abe_features = None
        if node_abe_features is not None:
            sorted_node_abe_features = node_abe_features[sorted_indices]
        sorted_block_idx = block_idx[sorted_indices]

        # Find the unique values and their first occurrence
        unique_block_idx, inverse_indices = torch.unique(sorted_block_idx, return_inverse=True, sorted=True)
        first_occurrence = torch.cat(
            [inverse_indices.new_zeros(1), (inverse_indices[1:] - inverse_indices[:-1]).nonzero(as_tuple=True)[0] + 1])
        for i in range(0, n_blocks, 2):
            start_x = first_occurrence[i]
            end_x = first_occurrence[i + 1]
            start_y = first_occurrence[i + 1]
            end_y = first_occurrence[i + 2] if i + 2 < n_blocks else len(data)
            # device = first_occurrence.device
            # edges_indices = torch.nonzero(
            #     torch.isin(edge_index, torch.arange(start_x, end_y, device=device)).all(dim=0)).squeeze()
            # edges = edge_index[:, edges_indices]
            # edges = edges - start_x
            # edges = torch.cat((edges, torch.flip(edges, dims=[0]),
            #                    torch.tensor([[0, start_y - start_x], [start_y - start_x, 0]])), dim=1)
            x = sorted_data[start_x:end_x]
            y = sorted_data[start_y:end_y]
            if hasattr(self, '_abs_pe_embedding'):
                node_abe_features_x = sorted_node_abe_features[start_x:end_x]
                node_abe_features_y = sorted_node_abe_features[start_y:end_y]
                positional_embedding_x = self._abs_pe_embedding(node_abe_features_x)
                positional_embedding_y = self._abs_pe_embedding(node_abe_features_y)
                x = x + positional_embedding_x
                y = y + positional_embedding_y
            attention_x, attention_y = compute_cross_attention(x, y, sim, self._node_state_dim)
            results.append(attention_x)
            results.append(attention_y)
        results = torch.cat(results, dim=0)

        # Revert back to the original order
        _, inverse_indices = torch.sort(sorted_indices)
        results = results[inverse_indices]

        return results


class GraphMatchingNet(GraphEmbeddingNet):
    """Graph matching net.

    This class uses graph matching layers instead of the simple graph prop layers.

    It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
    forms the first pair and graph 2 and 3 are the second pair etc., and computes
    cross-graph attention-based matching for each pair.
    """

    def __init__(self,
                 encoder,
                 aggregator,
                 node_state_dim,
                 edge_state_dim,
                 abs_pe_dim,
                 edge_hidden_sizes,
                 node_hidden_sizes,
                 n_prop_layers,
                 share_prop_params=False,
                 edge_net_init_scale=0.1,
                 node_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 abs_pe_embedding=False,
                 layer_class=GraphPropLayer,
                 similarity='dotproduct',
                 prop_type='embedding'):

        super(GraphMatchingNet, self).__init__(
            encoder,
            aggregator,
            node_state_dim,
            edge_state_dim,
            abs_pe_dim,
            edge_hidden_sizes,
            node_hidden_sizes,
            n_prop_layers,
            share_prop_params=share_prop_params,
            edge_net_init_scale=edge_net_init_scale,
            node_update_type=node_update_type,
            use_reverse_direction=use_reverse_direction,
            reverse_dir_param_different=reverse_dir_param_different,
            layer_norm=layer_norm,
            layer_class=GraphPropMatchingLayer,
            prop_type=prop_type,
            abs_pe_embedding=abs_pe_embedding
        )
        self._similarity = similarity

    def _apply_layer(self,
                     layer,
                     node_states,
                     from_idx,
                     to_idx,
                     graph_idx,
                     n_graphs,
                     edge_features,
                     **args,
                     ):
        node_abe_features = args['node_abe_features']
        """Apply one layer on the given inputs."""
        return layer(node_states=node_states, from_idx=from_idx, to_idx=to_idx, graph_idx=graph_idx, n_graphs=n_graphs,
                     similarity=self._similarity, edge_features=edge_features, node_features=None,
                     node_abe_features=node_abe_features, abs_pe_embedding=self._abs_pe_embedding)

    def _build_layer(self, layer_id):
        """Build one layer in the network."""
        return self._layer_class(
            self._node_state_dim,
            self._edge_state_dim,
            self._abs_pe_dim,
            self._edge_hidden_sizes,
            self._node_hidden_sizes,
            edge_net_init_scale=self._edge_net_init_scale,
            node_update_type=self._node_update_type,
            use_reverse_direction=self._use_reverse_direction,
            reverse_dir_param_different=self._reverse_dir_param_different,
            layer_norm=self._layer_norm,
            prop_type=self._prop_type,
            abs_pe_embedding=self._abs_pe_embedding
        )
        # name='graph-prop-%d' % layer_id)