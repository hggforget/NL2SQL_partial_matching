import collections
from GMN.graphembeddingnetwork import GraphEmbeddingNet, GraphEncoder, GraphAggregator
from GMN.graphmatchingnetwork import GraphMatchingNet
import torch

GraphData = collections.namedtuple('GraphData', [
    'from_idx',
    'to_idx',
    'node_features',
    'edge_features',
    'graph_idx',
    'n_graphs'])

def clean_df(df):
    df = df[(df['rel_path'] == 'relnode') & (df['label'] >= 0)]
    df = df[['G_rel', 'P_rel', 'db_id', 'label', 'answer_result', 'model_result', 'Prediction', 'Ground truth', 'UserPrompt', 'rel_score', 'type']]
    return df

def reshape_and_split_tensor(tensor, batch_size):

    """Reshape and split a 2D tensor along the last dimension.

        Args:
          tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
            multiple of `n_splits`.
          n_splits: int, number of splits to split the tensor into.

        Returns:
          splits: a list of `n_splits` tensors.  The first split is [tensor[0],
            tensor[n_splits], tensor[n_splits * 2], ...], the second split is
            [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
        """
    num_examples, feature_dim = tensor.shape
    # batch_size = num_examples // n_splits

    # Rearrange the tensor so that every n_splits items are adjacent.
    tensor = tensor.view(batch_size, 2, feature_dim).permute(1, 0, 2).contiguous()

    # Now, we can reshape the tensor back to its original shape and use torch.chunk to split it.
    tensor = tensor.view(num_examples, feature_dim)
    tensor_splits = torch.chunk(tensor, 2, dim=0)

    return tensor_splits


def build_model(config):
    """Create model for training and evaluation.

    Args:
      config: a dictionary of configs, like the one created by the
        `get_default_config` function.
      node_feature_dim: int, dimensionality of node features.
      edge_feature_dim: int, dimensionality of edge features.

    Returns:
      tensors: a (potentially nested) name => tensor dict.
      placeholders: a (potentially nested) name => tensor dict.
      AE_model: a GraphEmbeddingNet or GraphMatchingNet instance.

    Raises:
      ValueError: if the specified model or training settings are not supported.
    """

    encoder = GraphEncoder(**config['encoder'])
    aggregator = GraphAggregator(**config['aggregator'])
    if config['model_type'] == 'embedding':
        model = GraphEmbeddingNet(
            encoder, aggregator, **config['graph_embedding_net'])
    elif config['model_type'] == 'matching':
        model = GraphMatchingNet(
            encoder, aggregator, **config['graph_matching_net'])
    else:
        raise ValueError('Unknown model type: %s' % config['model_type'])

    optimizer = torch.optim.Adam((model.parameters()),
                                 lr=config['training']['learning_rate'], weight_decay=1e-5)

    return model, optimizer
