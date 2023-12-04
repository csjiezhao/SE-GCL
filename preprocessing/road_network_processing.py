import os
import math
import numpy as np
import osmnx as ox
import pandas as pd


def edge_attribute_statistics(df):
    for col in df.columns:
        pct_missing = np.mean(df[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing * 100)))


def edge_attribute_extraction(df, type2idx, save_path):
    """
    :param df: dataframe of road segment attributes
    :param type2idx: a dict, map "road_type" to label
    :param save_path:
    :return:
    """
    df['edge_idx'] = df['edge_idx'].map(eval)
    df.sort_values(by="edge_idx", inplace=True, ascending=True)

    '''attribute: length'''
    bin_size = 50
    df['length'] = df['length'].map(lambda x: math.floor(x / bin_size)).values.tolist()  # num_bins = 58

    '''attribute: lng and lat of midpoint'''
    polyline_lng = df.geometry.map(lambda x: [e[0] for e in x.coords]).values
    polyline_lng = sum(polyline_lng, [])
    polyline_lat = df.geometry.map(lambda x: [e[1] for e in x.coords]).values
    polyline_lat = sum(polyline_lat, [])
    lng0 = min(polyline_lng)
    lng1 = max(polyline_lng)
    lat0 = min(polyline_lat)
    lat1 = max(polyline_lat)

    coord_unit = 0.001  # about 100 meters
    num_lng_bins = int(np.ceil((lng1 - lng0) / coord_unit))
    lng_bins = [lng0 + i * coord_unit for i in range(num_lng_bins + 1)]
    num_lat_bins = int(np.ceil((lat1 - lat0) / coord_unit))
    lat_bins = [lat0 + i * coord_unit for i in range(num_lat_bins + 1)]

    df['mid_lng'] = df['midpoint'].map(lambda x: eval(x)[0])
    df['mid_lat'] = df['midpoint'].map(lambda x: eval(x)[1])
    df['mid_lng'] = pd.cut(df['mid_lng'], bins=lng_bins, labels=list(np.arange(num_lng_bins)))
    df['mid_lat'] = pd.cut(df['mid_lat'], bins=lat_bins, labels=list(np.arange(num_lat_bins)))

    '''select four attributes in total: LENGTH, ID, LNG, LAT'''
    segment_attr_code = df[['length', 'edge_idx', 'mid_lng', 'mid_lat']].values
    np.savez_compressed(os.path.join(save_path, 'segment_attr_code.npz'), data=segment_attr_code)

    '''extract road_type (i.e., the filed 'highway' in OSM) as label for [Road Type Classification]'''
    df['highway'] = df['highway'].map(lambda x: x[0] if type(x) == list else x)
    segment_type_label = df['highway'].map(type2idx).values
    np.savez_compressed(os.path.join(save_path, 'segment_type_label.npz'), data=segment_type_label)


if __name__ == '__main__':
    city_name = 'Porto'
    net_data_path = f'../data/{city_name}/road_network'

    # dict: used for converting road_type to numeric label
    highway2idx = {
        'primary': 0,
        'primary_link': 0,

        'secondary': 1,
        'secondary_link': 1,

        'tertiary': 2,
        'tertiary_link': 2,

        'residential': 3,

        'living_street': 4,

        'motorway': -1,
        'motorway_link': -1,
        'trunk': -1,
        'trunk_link': -1,
        'unclassified': -1,
        'road': -1,
        'busway': -1  # 2
    }

    G_file = os.path.join(net_data_path, f'{city_name}_G.graphml')  # origin graph
    G = ox.load_graphml(G_file)

    edge_attr_df = ox.graph_to_gdfs(G, nodes=False)
    edge_attr_df.reset_index(inplace=True)
    # edge_attribute_statistics(edge_attr_df)
    edge_attribute_extraction(edge_attr_df, highway2idx, save_path=net_data_path)
