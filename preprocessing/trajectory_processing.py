import os
import pandas as pd
import osmnx as ox
import json
import networkx as nx
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm


def raw_trajectory_clean(file):
    df = pd.read_csv(file, usecols=['TIMESTAMP', 'MISSING_DATA', 'POLYLINE'], index_col=None)
    df = df[~df['MISSING_DATA']]
    df.drop(columns=['MISSING_DATA'], inplace=True)
    df['POLYLINE'] = df['POLYLINE'].map(eval)
    df.columns = ['start_stamp', 'polyline']
    df['travel_time'] = df['polyline'].map(len) * 15
    df['end_stamp'] = df['start_stamp'] + df['travel_time']
    return df


def trajectory_matching(df):
    """Here you can employ the map matching algorithm according to your preferences,
       the returned df should add a new column named 'mpath', which represents the matched travel paths.
    """
    # - Fast Map Matching (FMM): https://fmm-wiki.github.io/
    # - ST-Matching: https://github.com/rottenivy/ST-Matching
    # - L2MM (our previous work): https://github.com/JiangLinLi/L2MM
    return df


def compute_segment_pass_time(row):
    start_stamp, polyline, travel_time, end_stamp, mpath = row
    timestamps = np.arange(start_stamp, end_stamp + 1, step=15)

    polyline = eval(polyline)
    mpath = eval(mpath)

    edges = [idx2edge[i] for i in mpath]
    nodes = [e[0] for e in edges] + [edges[-1][1]]
    node_pos = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in nodes]
    dist_mat = cdist(np.array(node_pos), np.array(polyline))
    min_idxs = dist_mat.argmin(axis=1)
    times = [timestamps[idx] for idx in min_idxs]
    pass_time = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
    return pass_time


def trajectory_feature_extraction(tra_df, save_path):
    segment_speed_dict = {}
    co_occur_mat = np.zeros((num_segments, num_segments), dtype=np.int32)

    for idx, row in tqdm(tra_df.iterrows()):
        mpath, pass_time = row
        mpath = eval(mpath)
        pass_time = eval(pass_time)

        for i, seg in enumerate(mpath):
            # compute segment speed
            if pass_time[i] > 0:
                seg_speed_temp = segment_length[seg] / pass_time[i]
                if seg_speed_temp < 30:
                    avg, n = segment_speed_dict.get(seg, (0, 0))
                    segment_speed_dict[seg] = ((avg * n + seg_speed_temp) / (n + 1), (n + 1))
            # compute oc_occur_mat
            for related_seg in mpath[i + 1:]:
                co_occur_mat[seg][related_seg] += 1

    np.savez_compressed(os.path.join(save_path, 'co_occur_mat.npz'), data=co_occur_mat)

    with open(os.path.join(save_path, 'segment_speed_dict.json'), "w", encoding='utf-8') as f2:
        json.dump(segment_speed_dict, f2, ensure_ascii=False, indent=4)
    segment_speed_arr = np.asarray([segment_speed_dict.get(str(i), (0, 0))[0] for i in range(num_segments)])
    np.savez_compressed(os.path.join(save_path, 'segment_speed_label.npz'), data=segment_speed_arr)


def travel_time_dataset_making(tra_df, padding_id, save_path):
    tra_df['mpath'] = tra_df['mpath'].map(eval)
    tra_df['path_len'] = tra_df['mpath'].map(len)

    '''generate time_est dataset'''
    min_len, max_len = 2, 99
    df1 = tra_df[(tra_df['path_len'] >= min_len) & (tra_df['path_len'] <= max_len)].copy()
    print(df1.shape)
    df1 = df1.iloc[:100000]  # 100k, 80k for training, 20k for testing
    print(df1.shape)

    num_samples = len(df1)
    x_arr = np.full([num_samples, max_len], padding_id, dtype=np.int32)
    x_len_arr = df1['path_len'].values
    y_arr = np.zeros([num_samples], dtype=np.float32)

    for i in tqdm(range(num_samples)):
        row = df1.iloc[i]
        path_arr = np.array(row['mpath'], dtype=np.int32)
        x_arr[i, :row['path_len']] = path_arr
        y_arr[i] = row['travel_time']

    np.savez_compressed(os.path.join(save_path, 'time_est_x'), data=x_arr)
    np.savez_compressed(os.path.join(save_path, 'time_est_x_len'), data=x_len_arr)
    np.savez_compressed(os.path.join(save_path, 'time_est_y'), data=y_arr)


if __name__ == '__main__':
    city_name = 'Porto'
    net_data_path = f'../data/{city_name}/road_network'
    traj_data_path = f'../data/{city_name}/trajectory'
    if not os.path.exists(traj_data_path):
        os.makedirs(traj_data_path)

    raw_traj_file = os.path.join(traj_data_path, 'train.csv')
    if not os.path.exists(raw_traj_file):
        raise FileNotFoundError(f'{city_name} Raw Trajectory File ("train.csv") Not Exist!')
    else:
        G_file = os.path.join(net_data_path, f'{city_name}_G.graphml')  # origin graph
        G = ox.load_graphml(G_file)
        num_segments = len(G.edges)
        segment_length = list(nx.get_edge_attributes(G, 'length').values())

        with open(os.path.join(net_data_path, f'{city_name}_idx2edge.json')) as f:
            idx2edge = json.load(f)
        idx2edge = {int(k): v for k, v in idx2edge.items()}

        traj_df = raw_trajectory_clean(raw_traj_file)  # start_stamp, polyline, travel_time, end_stamp
        traj_df = trajectory_matching(traj_df)  # start_stamp, polyline, travel_time, end_stamp, mpath
        traj_df['pass_time'] = traj_df.apply(compute_segment_pass_time, axis=1)

        traj_train_df = traj_df.iloc[:-360000]
        trajectory_feature_extraction(traj_train_df[['mpath', 'pass_time']], save_path=traj_data_path)
        task_df = traj_df.iloc[-360000:]
        travel_time_dataset_making(task_df, padding_id=num_segments, save_path=traj_data_path)





