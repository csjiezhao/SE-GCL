from model import SEGCL
from config import porto_config

import torch
import numpy as np
import random
from scipy import sparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_edge_removing_probability(trans_mat, adj_mat, epsilon):
    row_sum = np.sum(trans_mat, axis=1, keepdims=True)
    trans_mat = trans_mat / (row_sum + 1e-9)
    remove_prob = (1. - trans_mat) * adj_mat
    # rescale remove_prob into [epsilon, 1 - epsilon], epsilon is a very small number
    a = epsilon
    b = 1 - epsilon
    remove_prob = (a + (b - a) * remove_prob) * adj_mat
    return remove_prob


if __name__ == '__main__':

    # setup_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = porto_config
    sparse_adj = sparse.load_npz(params['adj_sp'])
    edge_index = from_scipy_sparse_matrix(sparse_adj)[0].to(device)

    attr_code = np.load(params['attr_code'])['data']
    vis_feat = np.load(params['vis_feat'])['data']
    co_occur_mat = np.load(params['co_occur_mat'])['data']
    adj = np.load(params['adj'])['data']
    local_trans_mat = adj * co_occur_mat
    edge_remove_prob = compute_edge_removing_probability(trans_mat=local_trans_mat,
                                                         adj_mat=adj,
                                                         epsilon=params['epsilon'])
    '''matrix used for negative sample selection'''
    neg_mat = co_occur_mat + co_occur_mat.T
    neg_mat = np.where(neg_mat > 0, 0, 1)

    neg_mat = torch.tensor(neg_mat).float().to(device)
    attr_code = torch.tensor(attr_code).long().to(device)
    vis_feat = torch.tensor(vis_feat).float().to(device)
    graph = Data(edge_index=edge_index,
                 node_attr=attr_code,
                 node_vis=vis_feat)

    edge_remove_prob = torch.tensor(edge_remove_prob).float().to(device)
    edge_remove_prob = edge_remove_prob[edge_index[0], edge_index[1]]

    '''stage 1: training'''
    marce = SEGCL(params=params).to(device)
    marce.train_process(g=graph, edge_remove_weights=edge_remove_prob, neg_mat=neg_mat)

    '''stage 2: readout the embeddings after training'''
    marce.eval()
    segment_emb, _ = marce(edge_index, attr_code, vis_feat)
    np.savez_compressed(f'embeddings/segcl_{params["city_name"]}_segment_emb.npz',
                        data=segment_emb.detach().cpu().numpy())
