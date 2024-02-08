from GMN.position_encoding.position_encoding import POSENCODINGS

def get_default_config():
    """The default configs."""
    model_type = 'matching'
    # Set to `embedding` to use the graph embedding net.
    node_state_dim = 32
    edge_state_dim = 16
    graph_rep_dim = 128
    abs_pe = 'rw'
    abs_pe_dim = 32
    abs_pe_embedding = False
    pe_type = 'whole'  # 'separate'
    graph_embedding_net_config = dict(
        abs_pe_dim=abs_pe_dim,
        node_state_dim=node_state_dim,
        edge_state_dim=edge_state_dim,
        edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],
        n_prop_layers=5,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here. gru
        node_update_type='gru',
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # set to True if your graph is directed
        reverse_dir_param_different=False,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
        # set to `embedding` to use the graph embedding net.
        prop_type=model_type)
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config['similarity'] = 'euclidean'  # other: euclidean, cosine
    graph_matching_net_config['abs_pe_embedding'] = abs_pe_embedding
    return dict(
        abs_pe=abs_pe,
        abs_pe_dim=abs_pe_dim,
        abs_pe_embedding=abs_pe_embedding,
        pe_type=pe_type,
        encoder=dict(
            node_hidden_sizes=[node_state_dim],
            edge_hidden_sizes=[edge_state_dim],
            node_feature_dim=64,
            edge_feature_dim = 1, #三种类型的edge
            res_out_channels=[128, 256, 512],
            res_num_features = 128,
            res_out_features = 32,
            res_num_blocks=[2,2,2,2],
            ),

        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],
            gated=True,
            aggregation_type='sum'),
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
        model_type=model_type,
        data=dict(
            problem='graph_edit_distance',
            dataset_params=dict(
                # always generate graphs with 20 nodes and p_edge=0.2.
                n_nodes_range=[20, 20],
                p_edge_range=[0.2, 0.2],
                n_changes_positive=1,
                n_changes_negative=2,
                validation_dataset_size=1000)),
        training=dict(
            batch_size=20,
            learning_rate=1e-4,
            mode='pair',
            loss='margin',  # other: hamming
            margin=0.4,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            # Increase this to train longer.
            n_training_steps=500000,
            # Print training information every this many training steps.
            print_after=10,
            # Evaluate on validation set every `eval_after * print_after` steps.
            eval_after=10),
        evaluation=dict(
            batch_size=20),
        seed=8,
    )
