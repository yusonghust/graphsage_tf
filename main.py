#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from collections import defaultdict
import time
import random
from config import cfg
from aggregator import *
import networkx as nx
import itertools as it
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
# tf.enable_eager_execution()

class graphsage():
    def __init__(self):
        self.cfg = cfg
        self.features = tf.Variable(self.cfg.features,dtype=tf.float32,trainable=False)
        if self.cfg.aggregator == 'mean':
            self.aggregator = mean_aggregator
        elif self.cfg.aggregator == 'pooling':
            self.aggregator = pooling_aggreagtor
        elif self.cfg.aggregator == 'lstm':
            self.aggregator = lstm_aggregator
        else:
            raise(Exception,"Invalid aggregator!")
        self.placeholders = self.build_placeholders()

    def build_placeholders(self):
        placeholders = {}
        if self.cfg.gcn:
            neigh_size = self.cfg.sample_num + 1
        else:
            neigh_size = self.cfg.sample_num
        placeholders['batchnodes']       = tf.placeholder(shape=(None),dtype=tf.int32)
        placeholders['samp_neighs_1st']  = tf.placeholder(shape=(None,neigh_size),dtype=tf.int32)
        if self.cfg.depth==2:
            placeholders['samp_neighs_2nd']  = tf.placeholder(shape=(None,neigh_size,neigh_size),dtype=tf.int32)
        if self.cfg.supervised:
            placeholders['labels'] = tf.placeholder(shape=(None),dtype=tf.int32)
        else:
            placeholders['input_1'] = tf.placeholder(shape=(None),dtype=tf.int32)
            placeholders['input_2'] = tf.placeholder(shape=(None),dtype=tf.int32)
            placeholders['input_3'] = tf.placeholder(shape=(None),dtype=tf.int32)
        return placeholders

    def construct_feed_dict_sup(self,nodes=None,samp_neighs_1st=None,samp_neighs_2nd=None,labels=None):
        feed_dict = {}
        feed_dict.update({self.placeholders['batchnodes']:nodes})
        feed_dict.update({self.placeholders['samp_neighs_1st']:samp_neighs_1st})
        feed_dict.update({self.placeholders['labels']:labels})
        if self.cfg.depth==2:
            feed_dict.update({self.placeholders['samp_neighs_2nd']:samp_neighs_2nd})
        return feed_dict

    def construct_feed_dict_unsup(self,nodes=None,samp_neighs_1st=None,samp_neighs_2nd=None,input_1=None,input_2=None,input_3=None):
        ###Note here labels are used for evaluate rather than training###
        feed_dict = {}
        feed_dict.update({self.placeholders['batchnodes']:nodes})
        feed_dict.update({self.placeholders['samp_neighs_1st']:samp_neighs_1st})
        feed_dict.update({self.placeholders['input_1']:input_1})
        feed_dict.update({self.placeholders['input_2']:input_2})
        feed_dict.update({self.placeholders['input_3']:input_3})
        if self.cfg.depth==2:
            feed_dict.update({self.placeholders['samp_neighs_2nd']:samp_neighs_2nd})
        return feed_dict

    def sample_neighs(self,nodes):
        _sample = np.random.choice
        neighs = [list(self.cfg.adj_lists[int(node)]) for node in nodes]
        samp_neighs = [list(_sample(neighs,self.cfg.sample_num,replace=False)) if len(neighs)>=self.cfg.sample_num else list(_sample(neighs,self.cfg.sample_num,replace=True)) for neighs in neighs]
        if self.cfg.gcn:
            samp_neighs = [samp_neigh+list([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        if self.cfg.aggregator=='lstm':
            # for lstm we need to shuffle the node order
            samp_neighs = [list(np.random.permutation(x)) for x in samp_neighs]
        return samp_neighs

    def forward(self):
        ### Here we set the aggregate depth as 2 ###
        if self.cfg.depth==2:
            agg_2nd = tf.map_fn(fn = lambda x:self.aggregator(tf.nn.embedding_lookup(self.features,x[0]),tf.nn.embedding_lookup(self.features,x[1]),self.cfg.dims,'agg_2nd'),
                elems=(self.placeholders['samp_neighs_1st'],self.placeholders['samp_neighs_2nd']),dtype=tf.float32)
            node_features = self.aggregator(tf.nn.embedding_lookup(self.features,self.placeholders['batchnodes']),tf.nn.embedding_lookup(self.features,self.placeholders['samp_neighs_1st']),self.cfg.dims,'agg_2nd')
            agg_1st = self.aggregator(node_features,agg_2nd,self.cfg.dims,'agg_1st')
        else:
            agg_1st = self.aggregator(tf.nn.embedding_lookup(self.features,self.placeholders['batchnodes']),tf.nn.embedding_lookup(self.features,self.placeholders['samp_neighs_1st']),
                self.cfg.dims,'agg_1st')
        return agg_1st

    def sess(self):
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=gpu_config)
        init = tf.global_variables_initializer()
        sess.run(init)
        return sess

    def supervised(self,inputs,labels):
        preds = tf.layers.dense(inputs,units=self.cfg.num_classes,activation=None)
        labels = tf.one_hot(labels,depth=self.cfg.num_classes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=preds)
        accuray = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds,1),tf.argmax(labels,1)),tf.float32))
        return loss,accuray

    def random_walk(self,num_walks=50,walk_length=4):
        G = nx.Graph()
        node_map = self.cfg.node_map
        with open(cfg.path + 'cora.cites','r') as f:
            for line in f:
                ls = line.strip().split()
                G.add_edge(node_map[ls[0]],node_map[ls[1]])
        f.close()
        nodes = list(G.nodes())
        degrees = [G.degree(x) for x in nodes]
        walk_pairs = []
        for n in nodes:
            if G.degree(n) == 0:
                continue
            for j in range(num_walks):
                current_n = n
                for k in range(walk_length+1):
                    neigs = list(G.neighbors(current_n))
                    if len(neigs)>0:
                        next_n = random.choice(neigs)
                    else:
                        break
                    if current_n != n:
                        walk_pairs.append((n,current_n))
                    current_n = next_n
        random.shuffle(walk_pairs)
        return walk_pairs,nodes,degrees

    def sample(self,pos_nodes,nodes,p):
        sample_nodes = []
        while len(sample_nodes)<self.cfg.neg_num:
            x = np.random.choice(nodes,size=1,replace=False,p=p)[0]
            if (x not in pos_nodes) and (x not in sample_nodes):
                sample_nodes.append(x)
        return sample_nodes

    def unsupervised(self,input_1,input_2,input_3):
        ###for unsupervised training, we use the loss function like deepwalk###
        aff = tf.reduce_sum(tf.multiply(input_1, input_2), 1)
        neg_aff = tf.matmul(input_1, tf.transpose(input_3))
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_mean(true_xent) + tf.reduce_mean(negative_xent)
        # loss = loss / tf.cast(tf.shape(input_1)[0], tf.float32)
        return loss

    def exec(self):
        if self.cfg.supervised:
            rand_indices = np.random.permutation(self.cfg.num_nodes)
            test = list(rand_indices[:1000])
            val = list(rand_indices[1000:1200])
            train = list(rand_indices[1200:])

            emb       = self.forward()
            loss,accu = self.supervised(emb,self.placeholders['labels'])
            opt       = tf.train.AdamOptimizer(self.cfg.lr).minimize(loss)
            sess      = self.sess()

            samp_neighs_1st = self.sample_neighs(val)
            samp_neighs_2nd = [self.sample_neighs(neighs) for neighs in samp_neighs_1st]
            val_label = self.cfg.labels[val]
            feed_dict_val = self.construct_feed_dict_sup(val,samp_neighs_1st,samp_neighs_2nd,val_label)

            for i in range(self.cfg.epochs):
                start = 0
                t = 0
                while start<len(train):
                    s = time.time()
                    end = min(start+self.cfg.batchsize,len(train))
                    batchnodes = train[start:end]
                    samp_neighs_1st = self.sample_neighs(batchnodes)
                    samp_neighs_2nd = [self.sample_neighs(neighs) for neighs in samp_neighs_1st]
                    train_label = self.cfg.labels[batchnodes]
                    feed_dict_train = self.construct_feed_dict_sup(batchnodes,samp_neighs_1st,samp_neighs_2nd,train_label)
                    _,ls_train,acc_train = sess.run([opt,loss,accu],feed_dict=feed_dict_train)
                    ls_val,acc_val = sess.run([loss,accu],feed_dict=feed_dict_val)
                    e = time.time()
                    t = t + e - s
                    print('\r Epoch = {:d} TrainLoss = {:.5f} TrainAccuracy = {:.3f} ValLoss = {:.5f} ValAccuracy = {:.3f} Time = {:.3f}'.format(i+1,ls_train,acc_train,ls_val,acc_val,t),end='\r')
                    start = end
                print()
            start = 0
            loss_list = []
            accu_list = []
            while start<len(test):
                end = min(start+self.cfg.batchsize,len(test))
                batchnodes = test[start:end]
                samp_neighs_1st = self.sample_neighs(batchnodes)
                samp_neighs_2nd = [self.sample_neighs(neighs) for neighs in samp_neighs_1st]
                test_label = self.cfg.labels[batchnodes]
                feed_dict_test = self.construct_feed_dict_sup(batchnodes,samp_neighs_1st,samp_neighs_2nd,test_label)
                ls_test,acc_test = sess.run([loss,accu],feed_dict=feed_dict_test)
                loss_list.append(ls_test*(end-start))
                accu_list.append(acc_test*(end-start))
                start = end
            print('TestLoss = ',sum(loss_list)/len(test),' TestAccuracy = ',sum(accu_list)/len(test))

        else:
            walk_pairs,nodes,degrees = self.random_walk()
            p = np.array(degrees)/sum(degrees)
            emb = self.forward()
            emb = tf.nn.l2_normalize(emb,1)
            input_1 = tf.nn.embedding_lookup(emb,self.placeholders['input_1'])
            input_2 = tf.nn.embedding_lookup(emb,self.placeholders['input_2'])
            input_3 = tf.nn.embedding_lookup(emb,self.placeholders['input_3'])
            loss = self.unsupervised(input_1,input_2,input_3)
            opt  = tf.train.GradientDescentOptimizer(self.cfg.lr).minimize(loss)
            sess = self.sess()
            for i in range(self.cfg.epochs):
                start = 0
                t = 0
                while start<len(walk_pairs):
                    s = time.time()
                    end = min(start+self.cfg.batchsize,len(walk_pairs))
                    batchpairs = walk_pairs[start:end]
                    input_1,input_2 = zip(*batchpairs)
                    input_1 = list(input_1)
                    input_2 = list(input_2)
                    input_3 = self.sample(input_2,nodes,p)
                    unique_nodes = list(set(input_1+input_2+input_3))
                    look_up = {x:i for i,x in enumerate(unique_nodes)}
                    samp_neighs_1st = self.sample_neighs(unique_nodes)
                    samp_neighs_2nd = [self.sample_neighs(neighs) for neighs in samp_neighs_1st]
                    input_1 = [look_up[x] for x in input_1]
                    input_2 = [look_up[x] for x in input_2]
                    input_3 = [look_up[x] for x in input_3]
                    feed_dict = self.construct_feed_dict_unsup(unique_nodes,samp_neighs_1st,samp_neighs_2nd,input_1,input_2,input_3)
                    _,ls = sess.run([opt,loss],feed_dict=feed_dict)
                    e = time.time()
                    t = t + e - s
                    print('\r Unsupervised Epoch = {:d} TrainLoss = {:.5f} Time = {:.3f}'.format(i+1,ls,t),end='\r')
                    start = end
                print()
            ### test ###
            start = 0
            embedding = np.zeros((self.cfg.num_nodes,self.cfg.dims))
            while(start<self.cfg.num_nodes):
                end = min(start + self.cfg.batchsize,self.cfg.num_nodes)
                unique_nodes = list(range(start,end))
                samp_neighs_1st = self.sample_neighs(unique_nodes)
                samp_neighs_2nd = [self.sample_neighs(neighs) for neighs in samp_neighs_1st]
                x = sess.run(emb,feed_dict={
                    self.placeholders['batchnodes']:unique_nodes,
                    self.placeholders['samp_neighs_1st']:samp_neighs_1st,
                    self.placeholders['samp_neighs_2nd']:samp_neighs_2nd
                    })
                embedding[unique_nodes] = x
                start = end
            print(embedding.shape)
            X,Y = [i for i in range(self.cfg.num_nodes)],[int(self.cfg.labels[i]) for i in range(self.cfg.num_nodes)]
            state = random.getstate()
            random.shuffle(X)
            random.setstate(state)
            random.shuffle(Y)
            index = int(self.cfg.num_nodes*self.cfg.clf_ratio)
            X_train = embedding[X[0:index]]
            Y_train = Y[0:index]
            X_test  = embedding[X[index:]]
            Y_test  = Y[index:]
            clf = LogisticRegression(random_state=0, solver='lbfgs',
                          multi_class='multinomial').fit(X_train,Y_train)
            print('TestAccuracy = ',clf.score(X_test,Y_test))



if __name__ == '__main__':
    sage = graphsage()
    sage.exec()