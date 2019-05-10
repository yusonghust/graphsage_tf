from collections import defaultdict
import numpy as np
def load_data(cfg):
    num_nodes = cfg.num_nodes
    num_feats = cfg.num_features
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open(cfg.path + 'cora.content') as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            # feat_data[i,:] = map(float, info[1:-1])
            feat_data[i,:] = [float(x) for x in info[1:-1]]
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open(cfg.path + 'cora.cites') as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists, node_map