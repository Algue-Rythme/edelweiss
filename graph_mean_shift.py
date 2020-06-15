import argparse
import collections
import itertools
import numpy as np
import numba
from sklearn.neighbors import NearestNeighbors
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
    parser.add_argument('--sim_measure', default='cosine', type=str, help='measure of similarity.')
    return parser.parse_args()

def get_grid_search(args):
    grid = GridSearch()
    for arg_name, arg_value in vars(args).items():
        grid.add_range(arg_name, [arg_value])
    # Algorithm
    grid.add_range('momentum', [0., 0.9, 0.95, 0.99])
    grid.add_range('stop_criterion', ['fixpoint'])
    grid.add_range('cautious', [True])
    # Similarity
    grid.add_range('lbda', [1.])
    grid.add_range('cube', [False])
    # Graph iterations
    grid.add_range('regular', [True])
    grid.add_range('kappa', [1.])
    grid.add_range('alpha', [0.])
    grid.add_range('bethoyd', [False])  # similarity path
    grid.add_range('bepon', [False])  # does not work
    grid.add_range('gripoyd', [False])  # conductance
    # Misc
    grid.add_range('verbose', [False])
    return grid

######################################
##### Edge Similarity Measures #######
######################################


def euclidian_similarity(params, x_latent):
    left = x_latent[np.newaxis,:,:]
    right = x_latent[:,np.newaxis,:]
    adj = left - right  # broadcasting
    adj = (adj**2).sum(axis=-1) # square of l2 norm
    adj = np.exp(-params.lbda * adj)
    return adj

def cosine_similarity(params, x_latent):
    adj = np.inner(x_latent, x_latent)
    adj = np.maximum(adj, 0.)  # remove negative entries
    if params.cube:
        adj = adj ** 3.
    return adj

def similarity_measure(params, x_latent):  # return positive edges only
    if params.sim_measure == 'euclidian':
        return euclidian_similarity(params, x_latent)
    if params.sim_measure == 'cosine':
        return cosine_similarity(params, x_latent)
    raise RuntimeError

######################################
########### Pre Processing ###########
######################################

def preprocessing(params, x_latent):
    if not params.denormalize_square:
        x_latent = x_latent ** 0.5
    if not params.denormalize_norm:
        norm = np.linalg.norm(x_latent, axis=1, keepdims=True)
        epsilon = 1e-3
        x_latent = x_latent / np.maximum(norm, epsilon)
    if not params.denormalize_mean:
        x_latent = x_latent - np.mean(x_latent, axis=0, keepdims=True)
    if not params.denormalize_norm:
        norm = np.linalg.norm(x_latent, axis=1, keepdims=True)
        epsilon = 1e-3
        x_latent = x_latent / np.maximum(norm, epsilon)
    return x_latent

@numba.jit(nopython=True, parallel=True)
def bethoyd_griwall(adj):
    num_nodes = adj.shape[0]
    for k in range(num_nodes):
        for i in numba.prange(num_nodes): #  pylint: disable=not-an-iterable
            for j in range(num_nodes):
                bridge = adj[i,k]*adj[k,j]
                adj[i,j] = max(adj[i,j], bridge)  # keep the path with better product

@numba.jit(nopython=True, parallel=True)
def bepon_warshoyd(adj):
    num_nodes = adj.shape[0]
    for k in range(num_nodes):
        for i in numba.prange(num_nodes): #  pylint: disable=not-an-iterable
            for j in range(num_nodes):
                bridge = min(adj[i,k], adj[k,j])
                adj[i,j] = max(adj[i,j], bridge)  # keep the path with the larger smallest edge

@numba.jit(nopython=True, parallel=True)
def inv_adj(matrix):
    n = matrix.shape[0]
    infty = float(1e6)
    for i in numba.prange(n): #  pylint: disable=not-an-iterable
        for j in range(n):
            if matrix[i,j] == 0.:
                matrix[i,j] = infty
            elif matrix[i,j] >= infty:
                matrix[i,j] = 0.
            else:
                matrix[i,j] = 1. / matrix[i,j]

@numba.jit(nopython=True, parallel=True)
def gripoyd_bethall(adj):
    inv_adj(adj)
    num_nodes = adj.shape[0]
    for k in range(num_nodes):
        for i in numba.prange(num_nodes): #  pylint: disable=not-an-iterable
            for j in range(num_nodes):
                bridge = adj[i,k] + adj[k,j]
                adj[i,j] = min(adj[i,j], bridge)  # keep the path with maximal conductance
    inv_adj(adj)

def transform_adjacency_matrix(params, adj):
    if params.sim_measure == 'euclidian':
        adj = adj + params.alpha*np.eye(N=adj.shape[0])
        adj = adj ** params.kappa
    if params.bethoyd:
        bethoyd_griwall(adj)
    if params.gripoyd:
        gripoyd_bethall(adj)
    if params.bepon:
        bepon_warshoyd(adj)
    return adj

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
    """Return a weighted adjacency list of heavier edges"""
    indexes = np.argsort(adj, axis=None)[::-1]
    num_nodes = int(adj.shape[0])
    max_edges = num_nodes * params.avg_degree
    neighbours = collections.defaultdict(list)
    cur_edge = 0
    for index in indexes:
        node_a, node_b = (index // num_nodes), (index % num_nodes)
        if params.regular and len(neighbours[node_a]) >= params.avg_degree:
            continue
        if adj[node_a, node_b] > 0.:
            cur_edge += 1
            neighbours[node_a].append(node_b)
        if cur_edge >= max_edges or adj[node_a, node_b] <= 0.:
            complete_missing_edges(params, neighbours, adj)
            break
    edges = [(i,j, adj[i,j]) for i, js in neighbours.items() for j in js]
    return edges

def create_default_labels(num_nodes, n_way):
    probs = np.full(shape=(num_nodes, n_way), fill_value=1./n_way)
    return probs

def graph_to_adjacency(graph):
    adj_lst = [[] for _ in range(graph.number_of_nodes())]
    for node in graph:
        for neighbor in graph.neighbors(node):
            adj_lst[node].append((neighbor, graph[node][neighbor]['weight']))
    return adj_lst

def get_orphans(graph, train_indexes):
    components = nx.connected_components(graph)
    comps = [set(map(int, component)) for component in components]
    train_indexes = set(train_indexes)
    orphans = [list(comp) for comp in comps if not comp & train_indexes]
    return orphans

def get_graph(params, x_latent, train_indexes):
    adj = similarity_measure(params, x_latent)
    adj = transform_adjacency_matrix(params, adj)
    edges_lst = edges_threshold(params, adj)  # keep heavier edges
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges_lst)
    orphans = get_orphans(graph, train_indexes)
    adj_lst = graph_to_adjacency(graph)
    probs = create_default_labels(int(x_latent.shape[0]), params.n_way)
    return adj_lst, probs, orphans


######################################
################ Misc ################
######################################

def check_accuracy(probs, labels, test_indexes):
    prediction = np.argmax(probs[test_indexes], axis=1)
    correct = prediction == labels[test_indexes]
    correct = np.array(correct).sum()
    acc = correct / len(test_indexes) * 100.
    return acc

def print_decision(probs, verbose):
    if not verbose:
        return
    prediction = np.argmax(probs, axis=1)
    for node in range(prediction.shape[0]):
        print('{%d,%d}'%(node, prediction[node]), end=' ')
    print('')

######################################
########### Graph Mean Shift #########
######################################

def stop_criterion(params, step, probs, com_increase_max, max_step):
    if step >= max_step:
        return True
    if params.stop_criterion == 'critical':
        return com_increase_max >= (params.n_shot + params.n_val)
    if params.stop_criterion == 'fixpoint':
        return False
    raise RuntimeError

def assign_supervised_labels(probs, labels, train_indexes):
    for index in train_indexes:
        label = labels[index]
        probs[index,:] = 0.
        probs[index,label] = 1.

def communities_proximity(params, adj_lst, probs):
    scores = np.zeros(shape=(len(adj_lst), params.n_way))
    for node, neighbors in enumerate(adj_lst): #  pylint: disable=not-an-iterable
        for neighbor, edge_weight in neighbors:
            for community in range(params.n_way):
                weight = edge_weight * probs[neighbor, community]
                scores[node, community] += weight  # threshold
    return scores

def get_com_increase_max(params, step, probs, decisions):
    com_increase_max = params.n_shot + params.n_val
    if step >= int(probs.shape[0]):
        return com_increase_max
    if params.cautious:
        com_increase_max = min(com_increase_max, params.n_shot + step + 1)
    return com_increase_max

@numba.jit(nopython=True)
def stable_matching(probs):
    num_nodes, num_coms = int(probs.shape[0]), int(probs.shape[1])
    com_size = num_nodes // num_coms
    taken = np.full(num_nodes, False)
    for com in range(num_coms):
        size = 0
        indexes = np.argsort(probs[:,com])[::-1]
        probs[:,com] = 0.
        for index in range(num_nodes):
            node = indexes[index]
            if not taken[node]:
                taken[node] = True
                probs[node,com] = 1.
                size += 1
                if size >= com_size:
                    break

def take_decision(transfer_prob):
    transfer_cpy = np.array(transfer_prob)
    stable_matching(transfer_cpy)
    decision = np.argmax(transfer_cpy, axis=1)
    return decision

def community_assignment(params, step, probs, scores):
    normalizer = np.sum(scores, axis=1, keepdims=True)
    transfer_prob = scores / (normalizer + 1.*(normalizer == 0))  # avoid division by 0
    decisions = take_decision(transfer_prob)  # non linearity, balanced by default
    com_increase_max = get_com_increase_max(params, step, probs, decisions)
    for community in range(params.n_way):
        com_indexes = np.nonzero(decisions == community)[0]
        com_probs = transfer_prob[com_indexes, community]
        score_indexes = np.argsort(com_probs)[::-1]  
        score_indexes = score_indexes[:com_increase_max] # keep the larger scores only, hoping no zeroes
        score_indexes = com_indexes[score_indexes]
        for index in score_indexes:
            a, b = params.momentum, 1. - params.momentum
            probs[index,:] *= a  # inertia :( :( :(
            probs[index, community] += b  # decision
    return com_increase_max

def assign_orphans(x_latent, probs, orphans):
    if not orphans:
        return
    orphans = [node for nodes in orphans for node in nodes]
    assigned = list(set(range(x_latent.shape[0])).difference(set(orphans)))
    x_train = x_latent[assigned,:]
    prediction = np.argmax(probs[assigned,:], axis=1)
    nn_train = np.zeros(shape=(probs.shape[1], x_latent.shape[1]))
    for com in range(int(probs.shape[1])):
        nn_train[com] = np.mean(x_train[prediction == com,:])
    x_test = x_latent[orphans,:]
    knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(nn_train)
    _, indices = knn.kneighbors(x_test)
    probs[orphans,:] = probs[indices[:,0],:]

def graph_mean_shift(params, x_latent, labels, train_indexes, test_indexes):
    x_latent = preprocessing(params, x_latent)  # project onto unit sphere
    adj_lst, probs, orphans = get_graph(params, x_latent, train_indexes)
    assign_supervised_labels(probs, labels, train_indexes)
    max_step = int(probs.shape[0] * 2.)
    progress = tqdm(total=max_step, leave=False, desc='')
    for step in itertools.count(0):  # infinite loop
        scores = communities_proximity(params, adj_lst, probs)
        com_increase_max = community_assignment(params, step, probs, scores)
        assign_supervised_labels(probs, labels, train_indexes)
        criterion = stop_criterion(params, step, probs, com_increase_max, max_step)
        if criterion:
            stable_matching(probs)
            assign_supervised_labels(probs, labels, train_indexes)
            assign_orphans(x_latent, probs, orphans)
        acc = check_accuracy(probs, labels, test_indexes)
        acc_desc = 'acc=%.2f%%'%acc
        progress.set_description(acc_desc)
        progress.update()
        if criterion:
            progress.close()
            print_decision(probs, params.verbose)
            return acc, acc_desc

def get_data(params):
    data_paths = {'vanilla':'images/latent/miniImagenet/ResNet/layer5/features.pt',
                  'yuqing':'images/latent/miniImagenet/WideResNet28_10_S2M2_R/last/novel.plk',
                  'cross':'images/latent/cross/WideResNet28_10_S2M2_R/last/output.plk',
                  'myriam-densenet-test':'images/latent/miniImagenet/DenseNet/test.pkl',
                  'myriam-wideresnet-test':'images/latent/miniImagenet/WideResNet-28-10/test.pkl',
                  'myriam-densenet-base':'images/latent/miniImagenet/DenseNet/train.pkl',
                  'myriam-wideresnet-base':'images/latent/miniImagenet/WideResNet-28-10/train.pkl'}
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
    progress.close()
    print('')

if __name__ == '__main__':
    args = parse_args()
    grid_search = get_grid_search(args)
    print(grid_search.get_constant_keys())
    for params in grid_search.get_params():
        print(grid_search.get_variable_keys(params))
        compute_stats(params)
