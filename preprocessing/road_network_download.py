import os
import osmnx as ox
import networkx as nx
import json
from scipy import sparse
import numpy as np

ox.config(use_cache=True, log_console=False)


def download_road_network(city: str,  boundary: list):
    if boundary:
        n, s, e, w = boundary
        g = ox.graph_from_bbox(north=n, south=s, east=e, west=w,
                               network_type='drive', simplify=True, truncate_by_edge=True)
    else:
        g = ox.graph_from_place(city, network_type='drive', simplify=True, truncate_by_edge=True)
    g.graph['name'] = f'{city}_G'
    return g


def create_edge_index(g):
    edge_list = list(g.edges)
    edge2idx = {e: i for i, e in enumerate(edge_list)}
    idx2edge = {i: e for i, e in enumerate(edge_list)}
    nx.set_edge_attributes(G, values=edge2idx, name='edge_idx')  # set edge indices to graph edge attribute
    return edge2idx, idx2edge


def export_graph_geo_data(g):
    node_gdf, edge_gdf = ox.graph_to_gdfs(g)
    node_gdf.reset_index(inplace=True)
    edge_gdf.reset_index(inplace=True)
    return node_gdf, edge_gdf


def convert_to_line_graph(g, edge2idx):
    lg = nx.line_graph(g)
    # note: the node attrs of L should be obtained from the edge attrs of G
    lg_node_attrs = {e: g.get_edge_data(*e) for e in g.edges}
    nx.set_node_attributes(lg, lg_node_attrs)  # update node attrs of L
    lg = nx.relabel.relabel_nodes(lg, edge2idx)  # use edge indices to relabel nodes of L
    # resort nodes of L according to edge index
    sorted_lg = nx.MultiDiGraph()
    sorted_lg.add_nodes_from(sorted(lg.nodes(data=True)))
    sorted_lg.add_edges_from(lg.edges(data=True))
    return sorted_lg


if __name__ == '__main__':

    city_name = 'Porto'

    net_data_path = f'../data/{city_name}/road_network'
    if not os.path.exists(net_data_path):
        os.makedirs(net_data_path)

    G_file = os.path.join(net_data_path, f'{city_name}_G.graphml')  # origin graph
    LG_file = os.path.join(net_data_path, f'{city_name}_LG.graphml')  # line graph

    if (not os.path.exists(G_file)) or (not os.path.exists(LG_file)):

        G = download_road_network(city=city_name, boundary=None)  # Nodes 5147 edges 10780

        '''create edge index (a unique identifier for each road segments)'''
        G_edge2idx, G_idx2edge = create_edge_index(G)
        with open(os.path.join(net_data_path, f'{city_name}_edge2idx.json'), "w", encoding='utf-8') as f:
            json.dump({str(k): v for k, v in G_edge2idx.items()}, f, ensure_ascii=False, indent=4)
        with open(os.path.join(net_data_path, f'{city_name}_idx2edge.json'), "w", encoding='utf-8') as f:
            json.dump(G_idx2edge, f, ensure_ascii=False, indent=4)

        '''export graph information'''
        G_node_gdf, G_edge_gdf = export_graph_geo_data(G)
        # to csv
        G_node_gdf.to_csv(os.path.join(net_data_path, f'{city_name}_G_nodes.csv'), index=False)
        G_edge_gdf.to_csv(os.path.join(net_data_path, f'{city_name}_G_edges.csv'), index=False)
        # to shp
        simplified_edge_gdf = G_edge_gdf[['u', 'v', 'key', 'edge_idx', 'geometry']]
        shp_path = os.path.join(net_data_path, 'shp')
        if not os.path.exists(shp_path):
            os.makedirs(shp_path)
        simplified_edge_gdf.to_file(os.path.join(shp_path, f'{city_name}_G_edges.shp'),
                                    driver='ESRI Shapefile', encoding='utf-8')

        '''calculate the first, mid and end points for all segments'''
        uvk = simplified_edge_gdf[['u', 'v', 'key']].apply(tuple, axis=1)
        first_points = simplified_edge_gdf['geometry'].map(lambda x: x.coords[0])
        last_points = simplified_edge_gdf['geometry'].map(lambda x: x.coords[1])
        midpoints = simplified_edge_gdf['geometry'].interpolate(0.5, normalized=True)
        midpoints = midpoints.map(lambda x: x.coords[0])

        edge2firstpoint = dict(zip(uvk, first_points))
        edge2lastpoint = dict(zip(uvk, last_points))
        edge2midpoint = dict(zip(uvk, midpoints))
        nx.set_edge_attributes(G, values=edge2firstpoint, name='p0')
        nx.set_edge_attributes(G, values=edge2lastpoint, name='p1')
        nx.set_edge_attributes(G, values=edge2midpoint, name='midpoint')

        '''save graph file'''
        ox.save_graphml(G, G_file)

        '''line graph transformation'''
        LG = convert_to_line_graph(G, G_edge2idx)
        LG.graph['name'] = f'{city_name}_LG'

        '''remove unless segment attributes'''
        for n, n_data in LG.nodes(data=True):
            for att in ['osmid', 'edge_idx', 'ref', 'bridge', 'name', 'access', 'tunnel', 'junction', 'area']:
                n_data.pop(att, None)

        '''export adjacency matrix of LG'''
        LG_adj_sp = nx.adjacency_matrix(LG)
        LG_adj = LG_adj_sp.todense()
        sparse.save_npz(os.path.join(net_data_path, f'{city_name}_LG_adj_sp'), LG_adj_sp)
        np.savez_compressed(os.path.join(net_data_path, f'{city_name}_LG_adj'), data=LG_adj)

        '''save line graph file'''
        ox.save_graphml(LG, LG_file)

        print(G)
        print(LG)
