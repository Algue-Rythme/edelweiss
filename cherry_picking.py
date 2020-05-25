import argparse
import collections
import itertools
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from tqdm import tqdm
import networkx as nx
from loaders import get_train_test_datasets
from utils import GridSearch


#################################
############ STUFF ##############
#################################

def process_history(cluster_history):
    histogram = collections.defaultdict(int)
    for changes in cluster_history.values():
        histogram[len(changes)] += 1
    return dict(sorted(list(histogram.items())))

def evaluate_bad_acc(probs, labels, cluster_history):
    bad_indexes = []
    for index, changes in cluster_history.items():
        if len(changes) > 5:
            bad_indexes.append(index)
    bad_indexes = set(bad_indexes)
    bad_indexes = [index for index in range(probs.shape[0]) if index not in bad_indexes]
    if not bad_indexes:
        return None
    return evaluate_acc(probs[bad_indexes,:], labels[bad_indexes])

#################################
########### ALGORITHM ###########
#################################

def preprocessing(params, x_latent):
    if not params.denormalize_square:
        x_latent = np.sqrt(x_latent)
    if not params.denormalize_mean:
        x_latent = x_latent - np.mean(x_latent, axis=0, keepdims=True)
    if not params.denormalize_norm:
        norm = np.linalg.norm(x_latent, axis=1, keepdims=True)
        epsilon = 1e-3
        x_latent = x_latent / np.maximum(norm, epsilon)
    return x_latent

def euclidian(params, x_latent, centroids):
    left = centroids[np.newaxis,:,:]
    right = x_latent[:,np.newaxis,:]
    adj = left - right  # broadcasting
    adj = (adj**2).sum(axis=-1) # square of l2 norm
    adj = np.exp(-params.lbda * adj)
    return adj  # shape=(N,K)

def cosine(params, x_latent, centroids):
    adj = np.inner(x_latent, centroids / np.linalg.norm(centroids, axis=1, keepdims=True))
    adj = np.maximum(adj, 0.)  # remove negative entries
    if params.cube:
        adj = adj ** 3.
    return adj

def get_dist(params, x_latent, centroids):
    if params.sim_measure == 'euclidian':
        return euclidian(params, x_latent, centroids)
    if params.sim_measure == 'cosine':
        return cosine(params, x_latent, centroids)
    raise ValueError

def oracle(params, probs, current_labels):
    for index, label in current_labels:
        probs[index,:] = 0.
        probs[index,label]= 1.

def default_labels(params, labels, shots):
    indexes = np.concatenate([way*params.n_shot + np.arange(shots) for way in range(params.n_way)])
    index_labels = [(index,labels[index]) for index in indexes]
    return index_labels

def evaluate_acc(probs, labels):
    decision = np.argmax(probs, axis=1)
    correct = np.sum(decision == labels)
    return correct * 100. / int(labels.shape[0])

def safe_normalize(array, axis):
    Z = np.sum(array, axis=axis, keepdims=True)
    return np.divide(array, Z, out=np.zeros_like(array), where=Z!=0)

def stable_matching(sim):
    num_nodes, num_coms = int(sim.shape[0]), int(sim.shape[1])
    com_size = num_nodes // num_coms
    taken = {com:com_size for com in range(num_coms)}
    nodes_indexes = list(range(num_nodes))
    nodes_indexes = sorted(nodes_indexes, # monotonic increasing order of nodes with highest similarity
                           key=(lambda index: tuple(np.sort(sim[index,:])[::-1])),
                           reverse=True)
    for index in nodes_indexes:
        coms = np.argsort(sim[index,:])[::-1]
        for com in coms:
            if taken[com] == 0:
                continue
            taken[com] -= 1
            sim[index,:] = 0.
            sim[index,com] = 1.
            break

def update_clusters_history(cluster_history, indexes, label):
    for index in indexes:
        if cluster_history[index][-1] == -1:
            cluster_history[index] = [label]
        elif label != cluster_history[index][-1]:
            cluster_history[index].append(index)

def mean_shift(params, x_latent, centroids, current_labels):
    cluster_history = {node:[-1] for node in range(x_latent.shape[0])}
    probs = np.zeros((x_latent.shape[0], params.n_way))
    for step in range(params.max_mean_shift_it):
        sim = get_dist(params, x_latent, centroids)
        sim = safe_normalize(sim, axis=1)
        if step+1 == params.max_mean_shift_it:
            if params.matching:
                stable_matching(sim)
        decisions = np.argmax(sim, axis=1)
        cluster_size = x_latent.shape[0]
        if step+1 != params.max_mean_shift_it:
            if not params.unbalanced:
                cluster_size = min([int(np.sum(decisions == label)) for label in np.arange(params.n_way)])
            if not params.uncautious:
                cluster_size = min(step + 1, cluster_size)
        probs.fill(0.)
        for label in range(params.n_way):
            indices_decision = np.nonzero(decisions == label)[0]
            scores = sim[indices_decision]
            indices_bigger = np.argsort(scores[:,label])[::-1]
            indices_bigger = indices_bigger[:cluster_size]
            indices_bigger = indices_decision[indices_bigger]
            probs[indices_bigger,:] = 0.
            probs[indices_bigger,label] = 1.
            update_clusters_history(cluster_history, indices_bigger, int(label))
        oracle(params, probs, current_labels)
        normalized_probs = safe_normalize(probs, 0)
        new_centroids = normalized_probs[:,np.newaxis,:] * x_latent[:,:,np.newaxis]
        new_centroids = np.transpose(np.sum(new_centroids, axis=0))
        mu = params.momentum if step < params.stop_momentum else 0.
        centroids = mu * centroids + (1. - mu) * new_centroids
    return probs, centroids, cluster_history

def k_means_initialization(params, x_latent):
    kmean = KMeans(n_clusters=params.n_way)
    kmean.fit(x_latent)
    centroids = np.array(kmean.cluster_centers_)
    return centroids

def supervised_initialization(params, x_latent, labels, shots):
    centroids = []
    for way in range(params.n_way):
        indexes = way * params.n_shot + np.arange(shots)
        centroids.append(np.mean(x_latent[indexes,:], axis=0))
    return np.stack(centroids, axis=0)

def initialization(params, x_latent, labels):
    if params.initialization == 'kmean':
        return k_means_initialization(params, x_latent)
    if params.initialization == 'supervised':
        return supervised_initialization(params, x_latent, labels, params.n_shot)
    if params.initialization == 'active':
        return supervised_initialization(params, x_latent, labels, 1)
    raise ValueError



def cherry_pick(params, x_latent, labels):
    current_labels = default_labels(params, labels, params.n_shot)
    x_latent = preprocessing(params, x_latent)
    centroids = initialization(params, x_latent, labels)
    probs, centroids, cluster_history = mean_shift(params, x_latent, centroids, current_labels)
    acc = evaluate_acc(probs, labels)
    return acc, cluster_history

######################################
############### Options ##############
######################################

def get_data(params):
    data_paths = {'vanilla':'images/latent/miniImagenet/ResNet/layer5/features.pt',
                  'yuqing':'images/latent/miniImagenet/WideResNet28_10_S2M2_R/last/novel.plk',
                  'cross':'images/latent/cross/WideResNet28_10_S2M2_R/last/output.plk'}
    n_way, n_shot, n_val = params.n_way, params.n_shot, params.n_val
    train_test = get_train_test_datasets(data_paths[params.data_path], n_way, n_shot, n_val)
    train_set, train_labels, test_set, test_labels = train_test
    x_latent = np.concatenate([train_set.numpy(), test_set.numpy()])
    labels = np.concatenate([train_labels.numpy(), test_labels.numpy()])
    return x_latent, labels

def compute_stats(params):
    accs = []
    if not params.no_progress:
        progress = tqdm(total=params.num_tests, leave=True, desc='')
    for _ in range(params.num_tests):
        x_latent, labels = get_data(params)
        acc, cluster_history = cherry_pick(params, x_latent, labels)
        accs.append(acc)
        desc = 'acc_avg=%.2f%%'%np.mean(accs)
        desc += ' acc=%.f%%'%acc
        if not params.no_progress:
            progress.set_description(desc)
            progress.update()
        else:
            print(desc)
    if not params.no_progress:
        progress.close()
    print('')

def parse_args():
    parser = argparse.ArgumentParser(description='Graph Mean Shift.')
    parser.add_argument('--data_path', default='yuqing', type=str, help='backbone.')
    parser.add_argument('--n_way', default=5, type=int, help='number of classes.')
    parser.add_argument('--n_val', default=15, type=int, help='number of validation examples.')
    parser.add_argument('--n_shot', default=5, type=int, help='number of training examples.')
    parser.add_argument('--num_tests', default=100, type=int, help='number of tests.')
    parser.add_argument('--max_mean_shift_it', default=100, type=int, help='number of tests.')
    parser.add_argument('--stop_momentum', default=50, type=int, help='number of tests.')
    parser.add_argument('--sim_measure', default='cosine', help='type of similarity')
    parser.add_argument('--initialization', default='supervised', help='initialization of the centroids')
    parser.add_argument('--denormalize_square', action='store_true')
    parser.add_argument('--denormalize_mean', action='store_true')
    parser.add_argument('--denormalize_norm', action='store_true')
    parser.add_argument('--lbda', default=10.)
    parser.add_argument('--cube', action='store_true')
    parser.add_argument('--uncautious', action='store_true')
    parser.add_argument('--unbalanced', action='store_true')
    parser.add_argument('--matching', action='store_true')
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--no_progress', action='store_true')
    return parser.parse_args()

def get_grid_search(args):
    grid = GridSearch()
    for arg_name, arg_value in vars(args).items():
        grid.add_range(arg_name, [arg_value])
    return grid

if __name__ == '__main__':
    args = parse_args()
    grid_search = get_grid_search(args)
    print(grid_search.get_constant_keys())
    for params in grid_search.get_params():
        print(grid_search.get_variable_keys(params))
        compute_stats(params)
