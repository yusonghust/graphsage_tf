#-*-coding:utf-8-*-
import numpy as np
import tensorflow as tf
import os
from utils import load_data

class config():
    def __init__(self):
        self._configs = {}

        self._configs['path']         = '/home/songyu/yu/graphsage-simple/cora/'
        self._configs['dims']         = 128
        self._configs['lr']           = 0.01
        self._configs['epochs']       = 10
        self._configs['num_nodes']    = 2708
        self._configs['num_features'] = 1433
        self._configs['num_classes']  = 7
        self._configs['sample_num']   = 10
        self._configs['clf_ratio']    = 0.5
        self._configs['batchsize']    = 10000
        self._configs['depth']        = 2 # 1 or 2
        self._configs['neg_num']      = 20 # negative sampling number for unsupervised training
        self._configs['act']          = tf.nn.relu
        self._configs['features']     = None
        self._configs['adj_lists']    = None
        self._configs['labels']       = None
        self._configs['node_map']     = None
        self._configs['gcn']          = True # whether add self-loop as gcn
        self._configs['concat']       = True # whether concat nodes with its neighbors
        self._configs['supervised']   = False # whether supervised training
        self._configs['aggregator']   = 'mean' # type of aggregators: mean,pooling,lstm

    @property
    def path(self):
        return self._configs['path']

    @property
    def dims(self):
        return self._configs['dims']

    @property
    def lr(self):
        return self._configs['lr']

    @property
    def epochs(self):
        return self._configs['epochs']

    @property
    def num_nodes(self):
        return self._configs['num_nodes']

    @property
    def num_features(self):
        return self._configs['num_features']

    @property
    def num_classes(self):
        return self._configs['num_classes']

    @property
    def features(self):
        return self._configs['features']

    @property
    def adj_lists(self):
        return self._configs['adj_lists']

    @property
    def labels(self):
        return self._configs['labels']

    @property
    def node_map(self):
        return self._configs['node_map']

    @property
    def sample_num(self):
        return self._configs['sample_num']

    @property
    def clf_ratio(self):
        return self._configs['clf_ratio']

    @property
    def batchsize(self):
        return self._configs['batchsize']

    @property
    def depth(self):
        return self._configs['depth']

    @property
    def neg_num(self):
        return self._configs['neg_num']

    @property
    def act(self):
        return self._configs['act']

    @property
    def gcn(self):
        return self._configs['gcn']

    @property
    def concat(self):
        return self._configs['concat']

    @property
    def supervised(self):
        return self._configs['supervised']

    @property
    def aggregator(self):
        return self._configs['aggregator']

    def update_config(self,key,value):
        if key in self._configs.keys():
            self._configs[key] = value
        else:
            raise RuntimeError('Update_Config_Error')

cfg = config()
feat_data, labels, adj_lists, node_map = load_data(cfg)
cfg.update_config('features',feat_data)
cfg.update_config('labels',labels)
cfg.update_config('adj_lists',adj_lists)
cfg.update_config('node_map',node_map)