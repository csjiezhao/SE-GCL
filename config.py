porto_config = {
    'city_name': 'Porto',
    'adj': 'data/Porto/road_network/Porto_LG_adj.npz',
    'adj_sp': 'data/Porto/road_network/Porto_LG_adj_sp.npz',
    'co_occur_mat': 'data/Porto/trajectory/co_occur_mat.npz',

    'attr_code': 'data/Porto/road_network/segment_attr_code.npz',
    'vis_feat': 'data/Porto/image/segment_vis_feat.npz',

    'num_seg_len': 44,  # bin numbers for segment length
    'num_seg_id': 10780,  # equals to the number of segments
    'num_seg_lng': 141,  # bin numbers for longitude
    'num_seg_lat': 49,  # bin numbers for longitude

    # params for multi-modal feature embedding module
    'seg_len_dim': 32,
    'seg_id_dim': 64,
    'seg_lng_dim': 16,
    'seg_lat_dim': 16,
    'vis_dim': 128,

    # params for graph encoder
    'hidden_dim': 128,
    'K': 3,
    'num_gcn_layers': 1,

    # params for projection head
    'latent_dim1': 64,
    'latent_dim2': 32,

    # params for edge removing
    'edge_dropout_ratio': 0.4,
    'epsilon': 0.1,  # a hyper-parameter for adjust edge removing probability

    'h': 1,
    'tau': 0.1,

    'lr': 0.01,
    'wd': 1e-5,
    'num_epochs': 1000,
}
