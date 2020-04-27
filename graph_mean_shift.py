import collections
import itertools
import os
import numpy as np
import scipy
import torch
import pygsp
import networkx as nx
from loaders import split_train_test
from loaders import get_train_test_datasets_labels
from loaders import get_dataset_from_datapath
from utils import GridSearch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Graph Mean Shift.')
    parser.add_argument('--n_way', default=5 type=int, help='number of classes.')
    parser.add_argument('--n_val', default=15, type=int, help='number of validation examples.')
    parser.add_argument('--n_shot', default=5, type=int, help='number of training examples.')
    parser.add_argument('--num_tests', default=1, type=int, help='number of tests.')
    parser.add_argument('--avg_degree', default=20, type=int, help='number of neighbors.')
    return parser.parse_args()

def get_grid_search(args):
    grid = GridSearch()
    for arg_name, arg_value in vars(args):
        grid.add_range(arg_name, [arg.value])
    grid.add_range('regular', [False])
    grid.add_range('cautious', [True])
    return grid

######################################
##### Edge Similarity Measures #######
######################################


def euclidian_similarity(x_latent, lbda):
    left = x_latent[np.newaxis,:,:]
    right = x_latent[:,np.newaxis,:]
    adj = left - right  # broadcasting
    adj = np.inner(adj, adj) # square of l2 norm
    adj = np.exp(-lbda * adj)
    return adj

######################################
########### Pre Processing ###########
######################################

def preprocessing(params, x_latent):
    x_latent = x_latent - np.mean(x_latent, axis=0, keepdims=True)
    norm = np.linalg.norm(x_latent, axis=1, keepdims=True)
    epsilon = 1e-3
    x_latent = x_latent / np.maximum(norm, epsilon)
    return x_latent

def transform_adjacency_matrix(params, adj):
    return adj

def transform_adjacency_list(params, adj_lst):
    return adj_lst

def complete_missing_edges(params, neighbors, adj):
    """complete adjacency list to avoid generate graphs"""
    min_neighbors = 2 # ensure 2 outgoing edges per node
    for node in neighbors:
        delta_neighbors = min_neighbors - len(neighbors[node])
        if delta_neighbors > 0:
            indexes = np.argsort(adj[node])[::-1]
            for index in indexes[len(neighbors[node]):min_neighbors]:
                neighbors[node].append(index)
            
def edges_threshold(params, adj):
    """Return a weighted adjacency list of heavier nodes"""
    indexes = np.argsort(adj, axis=None)[::-1]
    num_nodes, num_edges = int(adj.shape[0]), int(indexes.shape[0])
    max_edges = num_nodes * params.avg_degree
    neighbors = collections.defaultdict(list)
    cur_edge = 0
    for index in indexes:
        node_a, node_b = (index // num_nodes), (index % num_nodes)
        if regular and len(neighbours[a]) >= params.avg_degree:
            continue
        cur_edge += 1
        neighbours[node_a].append(node_b)
        if cur_edge >= max_edges:
            complete_missing_edges(neighbors, adj)
            break
    edges = [(i,j, adj[i,j]) for i, js in neighbors.items() for j in js]
    return edges

def assign_default_label(params, graph):
    sentinel_label = params.n_way+1 
    for node in graph:
        graph.add_node(node, label=sentinel_label)

def get_graph(params, x_latent):
    adj = euclidian_similarity(x_latent, lbda=10.)  # return positive edges only
    adj = transform_adjacency_matrix(params, adj)  # does nothing
    adj_lst = edges_threshold(params, weights)  # keep heavier edges
    adj_lst = transform_adjacency_list(params, adj_lst)  # does nothing
    graph = nx.Graph()
    graph.add_weighted_edges_from(adj_lst)
    assign_default_label(params, graph)
    return graph

def stop_criterion(step, graph):
    return step+1 >= 100

def assign_supervised_labels(graph, labels, train_indexes):
    for index in train_indexes:
        graph.add_node(index, label=None)

def communities_proximity(params, graph):
    scores = np.zeros(shape=(graph.number_of_nodes(), params.n_way+1))
    for node in graph:
        for neighbor in graph.neighbors():
            community = graph[neighbor]['label']
            if community is None:
                continue
            weight = graph[node, neighbor]['weight']
            scores[node, community] += weight
    return scores

def community_assignment(params, communities, scores, step):
    normalizer = np.sum(weights, axis=1, keepdims=True)
    normalizer[normalizer == 0] = 1.  # avoid division by 0
    probs = scores / normalizer
    decisions = np.argmax(probs, axis=1)
    smallest_com = np.min([(decisions == community).sum() for community in range(params.n_way)])
    if params.cautious:
        smallest_com = min(smallest_com, params.n_shot + step + 1)
    for community in range(params.n_way):
        com_indexes = decisions == community
        com_probs = probs[com_indexes, community]
        largest_score_indexes = np.argsort(com_probs)[:smallest_com-1:-1]  # keep the larger scores only
        largest_score_indexes = com_indexes[largest_score_indexes]
        for index in largest_score_indexes:
            graph.add_node(index, label=community)  # update label

def graph_mean_shift(params, x_latent, labels, train_indexes):
    x_latent = preprocessing(params, x_latent)  # project onto unit sphere
    graph = get_graph(params, x_latent)
    for step in itertools.count(0):
        assign_supervised_labels(graph, labels, train_indexes)
        communities, weights = communities_proximity(graph)
        cautious_community_assignment(params, communities, weights)
        if stop_criterion(step, graph):
            assign_supervised_labels(graph, labels, train_indexes)
            break

def get_data(params):
    data_path n_way, n_shot, n_val = params.data_path, params.n_way, params.n_shot, params.n_val
    train_test = get_train_test_datasets(data_path, n_way, n_shot, n_val)
    train_set, train_labels, test_set, test_labels = train_test
    x_latent = np.concatenate([train_set.numpy(), test_set.numpy()])
    labels = np.concatenate([train_labels.numpy(), test_labels.numpy()])
    train_indexes = list(range(int(train_labels.shape[0])))
    return x_latent, labels, train_indexes

def compute_stats(params):
    for _ in range(params.num_tests):
        x_latent, labels, train_indexes, test_indexes = get_data(params)
        predicted = graph_mean_shift(params, x_latent, labels, train_indexes)
        acc = compute_accuracy(labels, predicted, test_indexes)
        print(acc)

if __name__ == '__main__':
    args = parse_args()
    grid_search = get_grid_search(args)
    for params in grid_search.get_params():
        compute_stats(params)