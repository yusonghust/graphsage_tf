# Graphsage
Tensorflow implementation of ['Inductive Representation Learning on Large Graphs'](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs)   

## Introduction
A tensorflow implementation of graphsage, which is easier than the original implementation[GraphSAGE original implementation](https://github.com/williamleif/GraphSAGE).   
This code includes supervised and uinsupervised version, and three types of aggregators('mean','pooling' and 'lstm').   

## Requirement
python 3.6, tensorflow 1.12.0   

## Results
Here shows accuracy of the supervised and unsupervised graphsage with 'mean' aggregator.   

### The supervised graphsage accuracy is 0.871    

<div align=center><img src="https://github.com/cherisyu/graphsage/blob/master/sup.png" width="300" height="150" alt="supervised accuracy=0.871"/>       


### The unsupervised graphsage accuracy is 0.790    

<div align=center><img src="https://github.com/cherisyu/graphsage/blob/master/unsup.png" width="300" height="150" alt="unsupervised accuracy=0.0.79"/>   
