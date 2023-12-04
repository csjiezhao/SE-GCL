import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
from torch_geometric.utils import dropout_edge, to_networkx
from torch_geometric.data import Data
import torch.nn.functional as F
import networkx as nx
import time


class SEGCL(nn.Module):
    def __init__(self, params):
        super(SEGCL, self).__init__()
        self.params = params

        num_node_len = self.params['num_seg_len']
        num_node_id = self.params['num_seg_id']
        num_node_lng = self.params['num_seg_lng']
        num_node_lat = self.params['num_seg_lat']

        self.len_emb_layer = nn.Embedding(num_node_len + 1, self.params['seg_len_dim'], padding_idx=num_node_len)
        self.id_emb_layer = nn.Embedding(num_node_id + 1, self.params['seg_id_dim'], padding_idx=num_node_id)
        self.lng_emb_layer = nn.Embedding(num_node_lng + 1, self.params['seg_lng_dim'], padding_idx=num_node_lng)
        self.lat_emb_layer = nn.Embedding(num_node_lat + 1, self.params['seg_lat_dim'], padding_idx=num_node_lat)

        self.attr_dim = self.params['seg_len_dim'] + self.params['seg_id_dim'] + \
                        self.params['seg_lng_dim'] + self.params['seg_lat_dim']

        self.in_dim = self.attr_dim + self.params['vis_dim']

        self.graph_encoder = GraphEncoder(in_dim=self.in_dim,
                                          hidden_dim=self.params['hidden_dim'], cheb_k=self.params['K'],
                                          num_layers=self.params['num_gcn_layers'])

        self.projection_head = nn.Sequential(
            nn.Linear(self.params['hidden_dim'], self.params['latent_dim1']),
            nn.ELU(inplace=True),
            nn.Linear(self.params['latent_dim1'], self.params['latent_dim2'])
        )
        self.model_name = str(type(self).__name__)

    def forward(self, edge_index, node_attr, node_vis_feat):
        """extract node attr features"""
        node_len_feat = self.len_emb_layer(node_attr[:, 0])
        node_id_feat = self.id_emb_layer(node_attr[:, 1])
        node_lng_feat = self.lng_emb_layer(node_attr[:, 2])
        node_lat_feat = self.lat_emb_layer(node_attr[:, 3])

        node_features = torch.cat([node_id_feat, node_len_feat, node_lng_feat, node_lat_feat, node_vis_feat], dim=-1)

        h = self.graph_encoder(x=node_features, edge_index=edge_index)  # node embeddings for downstream tasks
        z = self.projection_head(h)  # node latent vectors for contrast
        return h, z

    @staticmethod
    def edge_removing(edge_index, ratio, remove_weights):
        if remove_weights is not None:
            retain_indices = torch.multinomial(input=1 - remove_weights,
                                               num_samples=int(edge_index.shape[1] * (1 - ratio)),
                                               replacement=False)
            edge_index = edge_index[:, retain_indices]
        else:
            edge_index, _ = dropout_edge(edge_index, p=ratio)
        return edge_index

    @staticmethod
    def h_hop_adjacency(network, h, device):
        num_nodes = len(network.nodes)
        h_hop_adj = torch.zeros((num_nodes, num_nodes), device=device)
        for i in range(num_nodes):
            h_neighs = nx.ego_graph(network, i, radius=h)
            for j in h_neighs.nodes:
                if j != i:
                    h_hop_adj[i, j] = 1.
        return h_hop_adj

    @staticmethod
    def similarity_computing(x1, x2):
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        return torch.matmul(x1, x2.t())

    def feature_masking(self, node_attr, node_vis):
        mask_choices = torch.randint(3, (node_attr.shape[0],))
        for i, m in enumerate(mask_choices):
            if m == 0:  # no_mask
                pass
            elif m == 1:  # attr_mask
                node_attr[i, 0], node_attr[i, 1], node_attr[i, 2], node_attr[i, 3] = \
                    self.params['num_seg_len'], self.params['num_seg_id'], \
                    self.params['num_seg_lng'], self.params['num_seg_lat']
            elif m == 2:  # vis_mask
                node_vis[i, :] = 0.
        return node_attr, node_vis

    def semi_loss(self, z1, z2, h_hop1, h_hop2, neg_mat):
        f = lambda x: torch.exp(x / self.params['tau'])
        intra_sim = f(self.similarity_computing(z1, z1))
        inter_sim = f(self.similarity_computing(z1, z2))

        intra_pos_sim = (intra_sim * h_hop1).sum(1)

        intra_neg = (1 - h_hop1 - torch.eye(z1.shape[0], device=z1.device))
        intra_neg = intra_neg * neg_mat
        intra_neg_sim = (intra_sim * intra_neg).sum(1)

        inter_pos_sim = inter_sim.diag()

        inter_neg = (1 - h_hop2 - torch.eye(z1.shape[0], device=z1.device))
        inter_neg = inter_neg * neg_mat
        inter_neg_sim = (inter_sim * inter_neg).sum(1)

        return -torch.log(
            (intra_pos_sim + inter_pos_sim)
            / (intra_neg_sim + inter_neg_sim))

    def contrastive_loss(self, z1, z2, h_hop1, h_hop2, neg_mat):
        loss1 = self.semi_loss(z1, z2, h_hop1, h_hop2, neg_mat)
        loss2 = self.semi_loss(z2, z1, h_hop2, h_hop1, neg_mat)
        loss = ((loss1 + loss2) * 0.5).mean()
        return loss

    def train_process(self, g, edge_remove_weights, neg_mat):
        print(f'Training SE-GCL model on {self.params["city_name"]} dataset ...')
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.params['lr'], weight_decay=self.params['wd'])
        self.train()
        best_loss = 1e9
        for e in range(1, self.params['num_epochs'] + 1):
            t0 = time.time()

            edge_index1 = g.edge_index.clone()
            edge_index2 = g.edge_index.clone()

            node_attr1, node_vis1 = g.node_attr.clone(), g.node_vis.clone()
            node_attr2, node_vis2 = g.node_attr.clone(), g.node_vis.clone()

            # remove edges by weights
            aug_edge_index1 = self.edge_removing(edge_index1, ratio=self.params['edge_dropout_ratio'],
                                                 remove_weights=edge_remove_weights)
            aug_edge_index2 = self.edge_removing(edge_index2, ratio=self.params['edge_dropout_ratio'],
                                                 remove_weights=edge_remove_weights)

            # mask features by modality
            aug_node_attr1, aug_node_vis1 = self.feature_masking(node_attr1, node_vis1)
            aug_node_attr2, aug_node_vis2 = self.feature_masking(node_attr2, node_vis2)

            node_emb1, node_vec1 = self.forward(aug_edge_index1, aug_node_attr1, aug_node_vis1)
            node_emb2, node_vec2 = self.forward(aug_edge_index2, aug_node_attr2, aug_node_vis2)

            # compute h-hop adjacency matrix of both graph views
            aug_g1 = to_networkx(Data(x=None, edge_index=edge_index1, num_nodes=g.num_nodes))
            aug_g2 = to_networkx(Data(x=None, edge_index=edge_index2, num_nodes=g.num_nodes))
            h_hop1 = self.h_hop_adjacency(network=aug_g1, h=self.params['h'], device=edge_index1.device)
            h_hop2 = self.h_hop_adjacency(network=aug_g2, h=self.params['h'], device=edge_index1.device)

            loss = self.contrastive_loss(node_vec1, node_vec2, h_hop1, h_hop2, neg_mat)
            print(f'Epoch: {e}, Loss: {loss.item()}, Time: {round(time.time() - t0, 5)}')

            if loss < best_loss:
                best_loss = loss
                self.save(epoch=e, lr=self.params['lr'])
            print('\n')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def save(self, epoch, lr):
        prefix = './checkpoints/'
        file_marker = self.model_name + '_lr' + str(lr) + '_e' + str(epoch) + '_' + self.params['city_name']
        model_path = time.strftime(prefix + '%m%d_%H_%M_' + file_marker + '.pth')
        torch.save(self.state_dict(), model_path)
        print('save parameters to file: %s' % model_path)

    def load(self, filepath, device):
        self.load_state_dict(torch.load(filepath, map_location=device))
        print('load parameters from file: %s' % filepath)


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, cheb_k, num_layers):
        super(GraphEncoder, self).__init__()
        self.graph_convs = nn.ModuleList([ChebConv(in_dim, hidden_dim, K=cheb_k)])
        self.graph_convs.extend([ChebConv(hidden_dim, hidden_dim, K=cheb_k)
                                 for _ in range(1, num_layers)])

    def forward(self, x, edge_index):
        for conv in self.graph_convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        return x
