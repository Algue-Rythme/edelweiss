import itertools
import sklearn.decomposition
import torch
import numpy as np
from latent_graph import embed_into_graph
from classifiers import train_logistic_regression
from models import get_device
from pygsp import graphs, filters, learning


def get_item(loss):
    return float(loss.item())

def symmetrize_edges(edge_loss_pairs):
    def reverse_edge(pair):
        loss, (a, b) = pair
        return loss, (b, a)
    symmetrics = map(reverse_edge, edge_loss_pairs)
    return edge_loss_pairs + list(symmetrics)

def filter_by_neighbours(num_nodes, all_edges, loss_per_edge, num_neighbors, regular):
    edge_loss_pairs = list(zip(loss_per_edge, all_edges))
    edge_loss_pairs = symmetrize_edges(edge_loss_pairs)
    edge_loss_pairs.sort(reverse=True)  # biggest similarity first
    max_edges = len(edge_loss_pairs) if regular else (num_neighbors * num_nodes)
    neighbours = [[] for _ in range(num_nodes)]
    threshold = 2 # non singular but not interesting
    for edge_idx, pair in enumerate(edge_loss_pairs):
        _, (a, _) = pair
        if regular and len(neighbours[a]) >= num_neighbors:
            continue
        if edge_idx < max_edges or len(neighbours[a]) < threshold:
            neighbours[a].append(pair)
    return itertools.chain.from_iterable(neighbours)

def get_degree(weights):
    return torch.sum(torch.abs(weights), dim=0)  # not suitable for undirected graphs

def laplacian_from_weights(weights):
    D = get_degree(weights)
    return D - weights

def edges_from_loss_fn(z_latent, loss_fn, num_neighbors, regular, normalize_weights=False, substract_mean=False):
    if substract_mean:
        z_latent = z_latent - torch.mean(z_latent, dim=0, keepdim=True)
    if normalize_weights:
        z_latent = z_latent / torch.norm(z_latent, dim=1, keepdim=True)
    num_nodes = int(z_latent.shape[0])
    all_edges = [(i, j) for i in range(num_nodes) for j in range(i)]
    loss_per_edge = loss_fn(z_latent, all_edges, reduce=False, memory_light=True)
    edge_loss_pairs = filter_by_neighbours(num_nodes, all_edges, loss_per_edge, num_neighbors, regular)
    return edge_loss_pairs

def weights_from_loss_fn(z_latent, loss_fn, num_neighbors, regular, undirected, normalize_weights=False):
    num_nodes = int(z_latent.shape[0])
    edge_loss_pairs = edges_from_loss_fn(z_latent, loss_fn, num_neighbors, regular, normalize_weights)
    weights = np.zeros(shape=(num_nodes, num_nodes), dtype=np.float32)
    for loss, (a, b) in edge_loss_pairs:
        weights[a, b] = loss
    weights = torch.tensor(weights)
    if undirected:
        weights = 0.5*weights + 0.5*torch.t(weights)  # symmetrize
    return weights

def get_adjacency_from_weights(weights):
    return (weights != 0).float()

def normalized_matrix(matrix, self_loops):
    num_nodes = int(matrix.shape[0])
    I = torch.eye(n=num_nodes)
    if self_loops:
        matrix = matrix + I
    D = get_degree(matrix)
    D_inv = torch.diag(torch.rsqrt(D))
    return torch.mm(torch.mm(D_inv, matrix), D_inv)

def simple_graph_convolution_matrix(weights, alpha, kappa):
    weights = normalized_matrix(weights, self_loops=True)
    I = torch.eye(n=int(weights.shape[0]))
    return torch.matrix_power(alpha*I + weights, kappa)

def get_diffusion_matrix(latent, loss_fn, params):
    weights = weights_from_loss_fn(latent, loss_fn, params.num_neighbors,
                                   regular=params.regular, undirected=params.undirected)
    weights = simple_graph_convolution_matrix(weights, params.alpha, params.kappa)
    return weights

def get_graph_and_support(model, x_latent, params):
    if params.latent_graph_support == params.latent_diffused == 'x_latent':
        return x_latent, x_latent
    z_latent = embed_into_graph(model, x_latent)
    latent_name = {'x_latent': x_latent, 'z_latent': z_latent}
    return latent_name[params.latent_graph_support], latent_name[params.latent_diffused]

def Combined_Yuqing_Myriam_Louis_Monster_Learnings(x_train, x_test, model, loss_fn, params):
    n_shot, n_val = int(x_train.shape[0]), int(x_test.shape[0])
    x_latent = torch.cat([x_train, x_test], dim=0)
    graph_support, latent_diffused =  get_graph_and_support(model, x_latent, params)
    diffusion = get_diffusion_matrix(graph_support, loss_fn, params)
    diffused = torch.mm(diffusion, latent_diffused)
    train_set, test_set = torch.split(diffused, split_size_or_sections=[n_shot, n_val], dim=0)
    return train_set, test_set

def Yuqing_Vanilla(x_latent, loss_fn, params):   
    diffusion = get_diffusion_matrix(x_latent, loss_fn, params)
    diffused = torch.mm(diffusion, x_latent)
    return diffused

def Pruned_Yuqing(recorder, x_train, x_test, labels_train, loss_fn, params):
    device = get_device()
    x_latent = torch.cat([x_train, x_test], dim=0)
    num_nodes = (params.n_shot + params.n_val) * params.n_way
    weights = weights_from_loss_fn(x_latent, loss_fn, params.num_neighbors,
                                   regular=params.regular, undirected=params.undirected)
    old_weights = torch.sum(weights)
    for _ in range(params.n_fixpoint_graph+1):
        diffusion = simple_graph_convolution_matrix(weights, params.alpha, params.kappa)
        diffused = torch.mm(diffusion, x_latent)
        classifier = train_logistic_regression(x_train, labels_train, params.n_way, device)
        with torch.no_grad():
            x_latent_deviced = x_latent.to(device)
            results = classifier(x_latent_deviced)
        _, predicted = torch.max(results.data, 1)
        predicted_l = predicted.repeat(num_nodes, 1)
        predicted_r = torch.t(predicted_l)
        mask = (predicted_l == predicted_r)
        mask = mask.float()
        weights = weights * mask.cpu()
    recorder.record_cut(torch.sum(weights / old_weights).item())
    return diffused

def combine_filters(filter_bank, filtered, params):
    # diffused = filter_bank.filter(filtered, method='chebyshev') # synthetize
    diffused = filtered[:,:,0] if len(filtered.shape) == 3 else filtered
    return diffused

def get_filter(params, graph):
    if params.filter_name == 'heat':
        return filters.Heat(G=graph, scale=params.frequency)
    elif params.filter_name == 'simoncelli':
        return filters.Simoncelli(G=graph, a=params.frequency)
    assert False

def fourrier_diffusion(x_latent, loss_fn, params):
    signal = x_latent.numpy()
    if params.svd_dim is not None:
        tsvd = sklearn.decomposition.TruncatedSVD(params.svd_dim)
        signal = tsvd.fit_transform(signal)
        assert signal.all() >= 0  # no negative components to avoid negative scalar products
    weights = weights_from_loss_fn(torch.FloatTensor(signal), loss_fn,
                                   params.num_neighbors, regular=params.regular,
                                   undirected=params.undirected, normalize_weights=False)
    graph = graphs.Graph(adjacency=weights.numpy().astype(np.float64))
    graph.estimate_lmax()
    filter_bank = get_filter(params, graph)
    filtered = filter_bank.filter(signal[:,:,np.newaxis], method='chebyshev')  # analyse -> N x S x F
    diffused = combine_filters(filter_bank, filtered, params)
    reconstructed = diffused
    if params.svd_dim is not None:
        reconstructed = tsvd.inverse_transform(reconstructed)
    reconstructed = torch.FloatTensor(reconstructed)
    error = torch.sum(torch.abs(x_latent - reconstructed) / torch.sum(torch.abs(x_latent)))  # relative error
    if params.svd_dim is not None and params.inverse_svd:
        assert reconstructed.all() >= 0
        return reconstructed, error
    return torch.FloatTensor(diffused), error

def tikhonov_diffusion(x_train, x_test, loss_fn, params):
    n_shot, n_val = int(x_train.shape[0]), int(x_test.shape[0])
    x_latent = torch.cat([x_train, x_test], dim=0)
    weights = weights_from_loss_fn(x_latent, loss_fn,
                                   params.num_neighbors, regular=True,
                                   undirected=True, normalize_weights=False)
    graph = graphs.Graph(adjacency=weights.numpy())
    mask = np.array([True]*n_shot + [False]*n_val)
    z_latent = learning.regression_tikhonov(graph, np.copy(x_latent.numpy()), mask, tau=1.)
    return torch.FloatTensor(z_latent)

def General_Yuqing(recorder, x_train, x_test, labels_train, loss_fn, params):
    n_val = int(x_test.shape[0])
    n_shot = int(x_train.shape[0])
    x_latent = torch.cat([x_train, x_test], dim=0)
    if params.transposed_diffusion:
        x_latent = torch.t(x_latent)
    if params.yuqing_version == 'vanilla':
        diffused = Yuqing_Vanilla(x_latent, loss_fn, params)
    elif params.yuqing_version == 'pruned':
        assert not params.transposed 
        diffused = Pruned_Yuqing(recorder, x_train, x_test, labels_train, loss_fn, params)
    elif params.yuqing_version == 'fourrier':
        diffused, error = fourrier_diffusion(x_latent, loss_fn, params)
        recorder.record_error(error)
    elif params.yuqing_version == 'tikhonov':
        diffused = tikhonov_diffusion(x_train, x_test, loss_fn, params)
    if params.transposed_diffusion:
        diffused = torch.t(diffused)
    num_train, num_test = params.n_shot * params.n_way, params.n_val * params.n_way
    train_set, test_set = torch.split(diffused, split_size_or_sections=[num_train, num_test], dim=0)
    return train_set, test_set