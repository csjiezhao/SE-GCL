from model import MARCE
from config import porto_config
from train import setup_seed

import torch
import numpy as np
from scipy import sparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix


if __name__ == '__main__':
    configs = [porto_config, chengdu_config]
    conf = configs[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sparse_adj = sparse.load_npz(conf['adj_sp'])
    edge_index = from_scipy_sparse_matrix(sparse_adj)[0].to(device)

    attr_code = np.load(conf['attr_code'])['data']
    vis_feat = np.load(conf['vis_feat'])['data']
    attr_code = torch.tensor(attr_code).long().to(device)
    vis_feat = torch.tensor(vis_feat).float().to(device)

    marce = MARCE(params=conf).to(device)
    marce.load('checkpoints/1124_04_36_MARCE_lr0.01_e970_porto.pth', device=device)
    marce.eval()
    segment_emb, _ = marce(edge_index, attr_code, vis_feat)
    np.savez_compressed(f'embeddings/marce_{conf["city_name"]}_segment_emb.npz', data=segment_emb.detach().cpu().numpy())