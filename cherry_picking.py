import argparse
import collections
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pointbiserialr, pearsonr
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import networkx as nx
from loaders_notorch import get_train_test_datasets


#################################
############ STUFF ##############
#################################

def diag_entropies(diag, mode):
    from gudhi import representations
    entropy = gudhi.representations.vector_methods.Entropy(mode=mode, resolution=10)
    return np.array(entropy(diag))

def TDA_monitoring(params, x_latent):
    import gudhi
    distance_matrix = np.arccos(np.clip(np.inner(x_latent,x_latent), -0.99, 0.99))
    rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=params.max_dimension)
    persistent = simplex_tree.persistence()
    diags = [simplex_tree.persistence_intervals_in_dimension(dim) for dim in range(params.max_dimension)]
    diags = [np.array([t for t in diag if not math.isinf(t[1])]) for diag in diags]
    if params.plot:
        gudhi.plot_persistence_diagram(persistent)
        plt.show()
    return diag_entropies(diags[0], mode='vector')

def print_densities(accs, instance_scores):
    persistent = np.array([t for i, instance in enumerate(instance_scores) if accs[i] >= 90. for t in instance])
    gudhi.plot_persistence_density(persistent)
    plt.show()

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

def update_clusters_history(cluster_history, indexes, label):
    for index in indexes:
        if cluster_history[index][-1] == -1:
            cluster_history[index] = [label]
        elif label != cluster_history[index][-1]:
            cluster_history[index].append(index)

def update_attribution_date(node_score, indexes, step):
    for index in indexes:
        if index not in node_score:
            node_score[index] = step

def reverse_node_score(node_score, probs, labels):
    indexes = list(node_score.keys())
    indexes.sort(key=(lambda k: node_score[k]))
    decision = np.argmax(probs, axis=1)
    correct = (decision == labels).astype(np.int64)
    return correct[indexes]

def get_dists(x_latent, centroids):
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    adj = ((centroids[np.newaxis,:,:] - x_latent[:,np.newaxis,:])**2).sum(axis=-1)  # broadcasting
    dists = np.min(adj, axis=1)
    node_score = {index:dists[index] for index in range(dists.shape[0])}
    return node_score

def get_entropy(x_latent, centroids):
    adj = ((centroids[np.newaxis,:,:] - x_latent[:,np.newaxis,:])**2).sum(axis=-1)  # broadcasting
    adj = safe_normalize(adj, axis=1)
    adj = np.maximum(adj, 1e-6)
    entropy = - np.sum(adj * np.log(adj), axis=1)
    node_score = {index:entropy[index] for index in range(x_latent.shape[0])}
    return node_score

def compute_instances_score(probs, centroids, node_score):
    instances_score = None
    return instances_score

#################################
########### ALGORITHM ###########
#################################

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

def karsher(params, x_latent, centroids):
    adj = np.inner(x_latent, centroids / np.linalg.norm(centroids, axis=1, keepdims=True))
    adj = np.clip(adj, -0.99, 0.99)
    num = np.arccos(adj)
    denom = np.sqrt(1. - adj**2)
    adj = num / denom
    adj = np.maximum(adj, 0.)  # remove negative entries
    return adj

def get_dist(params, x_latent, centroids):
    if params.sim_measure == 'euclidian':
        return euclidian(params, x_latent, centroids)
    if params.sim_measure == 'cosine':
        return cosine(params, x_latent, centroids)
    if params.sim_measure == 'karsher':
        return karsher(params, x_latent, centroids)
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

def normalize_sim(params, sim):
    if not params.sim_momentum:
        return safe_normalize(sim, axis=1)
    proximity = np.max(sim, axis=1, keepdims=True)
    normalized_sim = np.sum(sim, axis=1, keepdims=True)
    Z = (1. - proximity)*normalized_sim + proximity
    return np.divide(sim, Z, out=np.zeros_like(sim), where=Z!=0)

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

def karsher_centroids(params, centroids, x_latent, normalized_probs):
    for centroid_index in range(centroids.shape[0]):
        selected = normalized_probs[:,centroid_index] != 0.
        local_cluster = x_latent[selected,:]
        centroid = np.array(centroids[centroid_index,:])
        for _ in range(10):
            adj = np.inner(local_cluster, centroid / np.linalg.norm(centroid))  # cosine
            adj = np.clip(adj, -0.99, 0.99)
            num = np.arccos(adj)
            denom = np.sqrt(1. - adj**2)
            adj = num / denom
            adj = adj / np.sum(adj)  # normalize
            centroid = np.einsum('i,ij->j', adj, local_cluster)
        centroids[centroid_index,:] = centroid
    return centroids

def update_centroids(params, centroids, x_latent, normalized_probs, with_momentum=False):
    if params.karsher_centroids:
        return karsher_centroids(params, centroids, x_latent, normalized_probs)
    new_centroids = normalized_probs[:,np.newaxis,:] * x_latent[:,:,np.newaxis]
    new_centroids = np.transpose(np.sum(new_centroids, axis=0))
    mu = params.momentum if with_momentum else 0.
    centroids = mu * centroids + (1. - mu) * new_centroids
    return centroids

def mean_shift(params, x_latent, centroids, current_labels):
    node_score = dict()
    probs = np.zeros((x_latent.shape[0], params.n_way))
    for step in range(params.max_mean_shift_it):
        sim = get_dist(params, x_latent, centroids)
        if step+1 == params.max_mean_shift_it:
            if params.matching:
                stable_matching(sim)
        sim = normalize_sim(params, sim)
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
        oracle(params, probs, current_labels)
        normalized_probs = safe_normalize(probs, axis=0)
        centroids = update_centroids(params, centroids, x_latent, normalized_probs, with_momentum=(step<params.stop_momentum))
    return probs, centroids, node_score

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
    start_centroids = initialization(params, x_latent, labels)
    probs, centroids, node_score = mean_shift(params, x_latent, start_centroids, current_labels)
    acc = evaluate_acc(probs, labels)
    node_score = get_entropy(x_latent, centroids)
    node_score = reverse_node_score(node_score, probs, labels)
    instances_score = compute_instances_score(probs, centroids, node_score)
    return acc, node_score, instances_score

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
    x_latent = np.concatenate([train_set, test_set])
    labels = np.concatenate([train_labels, test_labels])
    return x_latent, labels

def update_desc(metric, vals, val, as_is=False, averaged=False, percentage=False):
    vals.append(val)
    symbol = '%' if percentage else ''
    if as_is:
        desc = '%s=%s%s'%(metric, str(val), symbol)
    else:
        desc = '%s=%.2f%s'%(metric, val, symbol)
    if averaged:
        if as_is:
            desc += ' %s_avg=%s%s'%(metric, str(np.mean(vals, axis=0)), symbol)
        else:
            desc += ' %s_avg=%.2f%s'%(metric, np.mean(vals), symbol)
    return desc

def init_plot(num_colors):
    if not params.progressive:
        return None, None
    plt.ion()
    fig, ax = plt.subplots()
    plt.xlim(0, 23)
    plt.ylim(0, 105)
    scs = [ax.scatter([], [], marker='.') for _ in range(num_colors)]
    fig.canvas.draw_idle()
    plt.pause(0.1)
    return fig, scs

def add_point(fig, scs, metric, accs):
    if not params.progressive:
        return
    metric = np.array(metric)
    for color, sc in enumerate(scs):
        sc.set_offsets(np.c_[metric[:,color], accs])
    fig.canvas.draw_idle()
    plt.pause(0.05)

def compute_stats(params):
    fig, scs = init_plot(params.max_dimension)
    accs = []
    node_scores = []
    instance_scores = []
    if not params.no_progress:
        progress = tqdm(total=params.num_tests, leave=True, desc='')
    for _ in range(params.num_tests):
        x_latent, labels = get_data(params)
        acc, node_score, instance_score = cherry_pick(params, x_latent, labels)
        node_scores.append(node_score)
        desc = update_desc('acc', accs, acc, averaged=True, percentage=True)
        desc += ' '+update_desc('instance_score', instance_scores, instance_score, averaged=False, as_is=True)
        # desc += ' '+update_desc('dates', node_scores, node_score, as_is=True)
        if not params.no_progress:
            progress.set_description(desc)
            progress.update()
        else:
            print(desc)
        add_point(fig, scs, instance_scores, accs)
    if not params.no_progress:
        progress.close()
    if params.progressive:
        fig.canvas.draw_idle()
    node_scores = np.array(node_scores)
    instance_scores = np.array(instance_scores)
    print('')
    print(np.mean(node_scores, axis=0))
    print('')
    # print_correlation('instances', accs, instance_scores)


def print_correlation(name, accs, metric):
    corrmatrix, p = pearsonr(accs, metric)
    print(name, 'c=%.3f'%corrmatrix, 'p=%.3f'%p)

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
    parser.add_argument('--momentum', default=1., type=float)
    parser.add_argument('--sim_momentum', action='store_true')
    parser.add_argument('--no_progress', action='store_true')
    parser.add_argument('--karsher_centroids', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--progressive', action='store_true')
    parser.add_argument('--max_dimension', default=2, type=int)
    return parser.parse_args()

def get_grid_search(args):
    dict_grid = {arg_name:[arg_value] for arg_name, arg_value in vars(args).items()}
    grid = ParameterGrid([dict_grid])
    Parameter = collections.namedtuple('Parameter', ' '.join(sorted(dict_grid.keys())))
    return grid, Parameter

if __name__ == '__main__':
    args = parse_args()
    grid_search, Parameter = get_grid_search(args)
    for dict_params in grid_search:
        params = Parameter(**dict_params)
        print(params)
        compute_stats(params)
