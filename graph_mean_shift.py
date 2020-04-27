import argparse
import collections
import itertools
import numpy as np
from tqdm import tqdm
import networkx as nx
from loaders import get_train_test_datasets
from utils import GridSearch


def parse_args():
    parser = argparse.ArgumentParser(description='Graph Mean Shift.')
    parser.add_argument('--data_path', default='yuqing', type=str, help='backbone.')
    parser.add_argument('--n_way', default=5, type=int, help='number of classes.')
    parser.add_argument('--n_val', default=15, type=int, help='number of validation examples.')
    parser.add_argument('--n_shot', default=5, type=int, help='number of training examples.')
    parser.add_argument('--num_tests', default=100, type=int, help='number of tests.')
    parser.add_argument('--avg_degree', default=20, type=int, help='number of neighbors.')
    parser.add_argument('--sim_measure', default='euclidian', type=str, help='measure of similarity.')
    return parser.parse_args()

def get_grid_search(args):
    grid = GridSearch()
    for arg_name, arg_value in vars(args).items():
        grid.add_range(arg_name, [arg_value])
    grid.add_range('regular', [False])
    grid.add_range('cautious', [True])
    grid.add_range('stop_criterion', ['critical'])
    return grid

######################################
##### Edge Similarity Measures #######
######################################


def euclidian_similarity(x_latent, lbda):
    left = x_latent[np.newaxis,:,:]
    right = x_latent[:,np.newaxis,:]
    adj = left - right  # broadcasting
    adj = (adj**2).sum(axis=-1) # square of l2 norm
    adj = np.exp(-lbda * adj)
    return adj

def cosine_similarity(x_latent):
    adj = np.inner(x_latent, x_latent)
    return adj

def similarity_measure(params, x_latent):
    if params.sim_measure == 'euclidian':
        return euclidian_similarity(x_latent, lbda=10.)  # return positive edges only
    if params.sim_measure == 'cosine':
        return cosine_similarity(x_latent)
    raise RuntimeError

######################################
########### Pre Processing ###########
######################################

def preprocessing(params, x_latent):
    # Magic trick 1
    x_latent = np.power(x_latent, 0.5)
    # Magic trick 2
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
    num_nodes = int(adj.shape[0])
    max_edges = num_nodes * params.avg_degree
    neighbours = collections.defaultdict(list)
    cur_edge = 0
    for index in indexes:
        node_a, node_b = (index // num_nodes), (index % num_nodes)
        if params.regular and len(neighbours[node_a]) >= params.avg_degree:
            continue
        cur_edge += 1
        neighbours[node_a].append(node_b)
        if cur_edge >= max_edges:
            complete_missing_edges(params, neighbours, adj)
            break
    edges = [(i,j, adj[i,j]) for i, js in neighbours.items() for j in js]
    return edges

def assign_default_label(graph):
    for node in graph:
        graph.add_node(node, label=None)

def get_graph(params, x_latent):
    adj = similarity_measure(params, x_latent)
    adj = transform_adjacency_matrix(params, adj)  # does nothing
    adj_lst = edges_threshold(params, adj)  # keep heavier edges
    adj_lst = transform_adjacency_list(params, adj_lst)  # does nothing
    graph = nx.Graph()
    graph.add_weighted_edges_from(adj_lst)
    assign_default_label(graph)
    return graph

def stop_criterion(params, step, graph, com_increase_max):
    if params.stop_criterion == 'critical':
        max_step = graph.number_of_nodes()
        if step >= max_step:
            return True
        return com_increase_max >= (params.n_shot + params.n_val)
    if params.stop_criterion == 'fix point':
        return False
    raise RuntimeError

def assign_supervised_labels(graph, labels, train_indexes):
    for index in train_indexes:
        graph.add_node(index, label=labels[index])

def communities_proximity(params, graph):
    scores = np.zeros(shape=(graph.number_of_nodes(), params.n_way+1))
    for node in graph:
        for neighbor in graph.neighbors(node):
            community = graph.nodes[neighbor]['label']
            if community is None:
                continue
            weight = graph[node][neighbor]['weight']
            scores[node, community] += weight
    return scores

def get_com_increase_max(params, step, graph, decisions):
    if step >= graph.number_of_nodes():
        return graph.number_of_nodes()
    smallest_com = np.min([(decisions == community).sum() for community in range(params.n_way)])
    if params.cautious:
        smallest_com = min(smallest_com, params.n_shot + step + 1)
    return smallest_com

def community_assignment(params, step, graph, scores):
    normalizer = np.sum(scores, axis=1, keepdims=True)
    probs = scores / (normalizer + 1.*(normalizer == 0))  # avoid division by 0
    decisions = np.argmax(probs, axis=1)
    com_increase_max = get_com_increase_max(params, step, graph, decisions)
    assign_default_label(graph)
    for community in range(params.n_way):
        com_indexes = np.nonzero(decisions == community)[0]
        com_probs = probs[com_indexes, community]
        score_indexes = np.argsort(com_probs)[::-1]  
        score_indexes = score_indexes[:com_increase_max] # keep the larger scores only, hoping no zeroes
        score_indexes = com_indexes[score_indexes]
        for index in score_indexes:
            graph.add_node(index, label=community)  # update label
    return com_increase_max

def check_accuracy(graph, labels, test_indexes):
    correct = [graph.nodes[node]['label'] == labels[node] for node in test_indexes]
    correct = np.array(correct).sum()
    acc = correct / len(test_indexes) * 100.
    return acc

def print_decision(graph, verbose=0):
    if not verbose:
        return
    for node in sorted(list(graph)):
        print('{%d,%d}'%(node, graph.nodes[node]['label']), end=' ')
    print('')

def graph_mean_shift(params, x_latent, labels, train_indexes, test_indexes):
    x_latent = preprocessing(params, x_latent)  # project onto unit sphere
    graph = get_graph(params, x_latent)
    assign_supervised_labels(graph, labels, train_indexes)
    progress = tqdm(total=graph.number_of_nodes()+1, leave=False, desc='')
    for step in itertools.count(0):  # infinite loop
        scores = communities_proximity(params, graph)
        com_increase_max = community_assignment(params, step, graph, scores)
        assign_supervised_labels(graph, labels, train_indexes)
        acc = check_accuracy(graph, labels, test_indexes)
        acc_desc = 'acc=%.2f%%'%acc
        progress.set_description(acc_desc)
        progress.update()
        if stop_criterion(params, step, graph, com_increase_max):
            progress.close()
            print_decision(graph)
            desc = 'error=%d '%sum([(graph.nodes[node]['label'] is None) for node in graph])
            return acc, desc+acc_desc

def get_data(params):
    data_paths = {'vanilla':'images/latent/miniImagenet/ResNet/layer5/features.pt',
                  'yuqing':'images/latent/miniImagenet/WideResNet28_10_S2M2_R/last/novel.plk',
                  'cross':'images/latent/cross/WideResNet28_10_S2M2_R/last/output.plk'}
    n_way, n_shot, n_val = params.n_way, params.n_shot, params.n_val
    train_test = get_train_test_datasets(data_paths[params.data_path], n_way, n_shot, n_val)
    train_set, train_labels, test_set, test_labels = train_test
    x_latent = np.concatenate([train_set.numpy(), test_set.numpy()])
    labels = np.concatenate([train_labels.numpy(), test_labels.numpy()])
    train_indexes = list(range(int(train_labels.shape[0])))
    max_node = int(x_latent.shape[0])  # pylint: disable=E1136  # pylint/issues/3139
    test_indexes = list(range(int(train_labels.shape[0]), max_node))
    return x_latent, labels, train_indexes, test_indexes

def compute_stats(params):
    accs = []
    progress = tqdm(total=params.num_tests, leave=True, desc='')
    for _ in range(params.num_tests):
        x_latent, labels, train_indexes, test_indexes = get_data(params)
        acc, desc = graph_mean_shift(params, x_latent, labels, train_indexes, test_indexes)
        accs.append(acc)
        avg_acc = np.mean(accs)
        progress.set_description(desc+' acc_avg=%.2f%%'%avg_acc)
        progress.update()

if __name__ == '__main__':
    args = parse_args()
    grid_search = get_grid_search(args)
    for params in grid_search.get_params():
        compute_stats(params)
