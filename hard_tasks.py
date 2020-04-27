import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
import functools
import pickle


def get_labels_stats(labels):
    labels_set = torch.unique(labels).numpy().tolist()
    num_labels = len(labels_set)
    n_sample_per_label = labels.shape[0] // num_labels
    return num_labels, n_sample_per_label

def data_subset(data, labels, n_way, ways=None):
    # WARNING: ASSUME CONTIGUOUS LABELS IN EQUAL NUMBER
    num_labels, n_sample_per_label = get_labels_stats(labels)
    if ways is None:
        ways = np.random.choice(range(num_labels), n_way, replace=False)
    subset_data = []
    subset_labels = []
    for new_l, l in enumerate(ways):
        start_sample, end_sample = l*n_sample_per_label, (l+1)*n_sample_per_label
        subset_data.append(data[start_sample : end_sample])
        subset_labels.append(torch.LongTensor([new_l]*n_sample_per_label))  # RELABELLING
    subset_data = torch.cat(subset_data, dim=0)
    subset_labels = torch.cat(subset_labels, dim=0)
    return subset_data, subset_labels

def extract_from_slice(cur_data_slice, select_train, select_test, train_set, test_set):
    cur_train = cur_data_slice[select_train]
    cur_test = cur_data_slice[select_test]
    train_set.append(cur_train)
    test_set.append(cur_test)

def split_train_test(data, labels, n_shot, n_val):
    num_labels, n_sample_per_label = get_labels_stats(labels)
    assert n_shot + n_val <= n_sample_per_label
    train_set, test_set  = [], []
    train_labels, test_labels = [], []
    for l in range(num_labels):
        select_elems = np.random.choice(range(n_sample_per_label), n_shot + n_val, replace=False)
        select_train = select_elems[:n_shot]
        select_test = select_elems[n_shot:]
        start_sample = l * n_sample_per_label
        end_sample = start_sample + n_sample_per_label
        extract_from_slice(data[start_sample:end_sample], select_train, select_test, train_set, test_set)
        extract_from_slice(labels[start_sample:end_sample], select_train, select_test, train_labels, test_labels)
    train_set = torch.cat(train_set, dim=0)
    test_set = torch.cat(test_set, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    return train_set, train_labels, test_set, test_labels

def load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        labels = [np.full(shape=len(data[key]), fill_value=key) for key in data]
        data = [features for key in data for features in data[key]]
        dataset = dict()
        dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
        dataset['labels'] = torch.LongTensor(np.concatenate(labels))
        return dataset

@functools.lru_cache()
def get_dataset_from_datapath(data_path):
    if data_path.endswith('.plk'):
        dataset = load_pickle(data_path)
    elif data_path.endswith('.pt'):
        dataset = torch.load(data_path)
    else:
        assert False
    original_data = dataset['data']
    original_labels = dataset['labels']
    return original_data, original_labels

def get_train_test_datasets_labels(data_path, n_way, n_shot, n_val, ways=None):
    original_data, original_labels = get_dataset_from_datapath(data_path)
    num_labels, n_sample_per_label = get_labels_stats(original_labels)
    if ways is None:
        ways = np.random.choice(range(num_labels), n_way, replace=False)
    selected_labels = [int(original_labels[i*n_sample_per_label]) for i in ways]
    data, labels = data_subset(original_data, original_labels, n_way, ways=ways)
    return split_train_test(data, labels, n_shot, n_val), selected_labels

def gather_edges(graph, ways):
    edges = []
    for str_edge in graph.edges.data('weight'):
        edge = int(str_edge[0]), int(str_edge[1]), float(str_edge[2])
        if edge[0] in ways and edge[1] in ways:
            edges.append(edge)
    return edges

def get_subgraph_weight(edges):
    return sum([edge[2] for edge in edges])

def get_worse_clusters(n_way, big_graph, data_path):
    _, original_labels = get_dataset_from_datapath(data_path)
    num_labels, n_sample_per_label = get_labels_stats(original_labels)
    combinations_iter = itertools.combinations(list(range(big_graph.number_of_nodes())), n_way)
    combinations = []
    for i, ways in enumerate(combinations_iter):
        real_ways = [int(original_labels[i*n_sample_per_label]) for i in ways]
        combinations.append((get_subgraph_weight(gather_edges(big_graph, real_ways)), ways))
    combinations.sort(key=(lambda t: t[0]), reverse=True)
    return combinations

def hard_tasks(data_path, dot_name, n_way, n_shot, n_val, plot_graph=False):
    # the dot file is used to load the worse tasks
    dot_graph = nx.drawing.nx_agraph.read_dot(dot_name)
    edges = [(int(str_edge[0]), int(str_edge[1]), float(str_edge[2])) for str_edge in dot_graph.edges.data('weight')]
    big_graph = nx.Graph()
    big_graph.add_weighted_edges_from(edges)
    if plot_graph:
        pos = nx.spring_layout(big_graph)
        options, edge_labels = get_draw_options(big_graph)
        nx.draw(big_graph, pos=pos, **options)
        nx.draw_networkx_edge_labels(big_graph, edge_labels=edge_labels, pos=pos, font_size=8)
        plt.show()
    # the graph built from it can now be used to retrieve the worst combinations of classes
    clusters = get_worse_clusters(n_way, big_graph, data_path)
    # TODO(@Myriam): now you can iterate over clusters
    # By default clusters are sorted in decreasing weight order
    # The bigger the weight, the bigger the confusion
    # For example, you can do:
    #       clusters = clusters[:100]
    # to only keep 100 worse of them
    # or:
    #       clusters[random.randint(10, 20)]
    # to sample a cluster randomly between the 10th and the 20th
    for repet in [0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1]: # here I iterate in order, over the five first and the five last
        # retrieve the labels indexes
        fake_ways = clusters[repet][1]
        # then retrieve train set, test set, and the real value of the labels (starting at 80 for wideresnet):
        train_test, ways = get_train_test_datasets_labels(data_path, n_way, n_shot, n_val, ways=fake_ways)
        train_set, train_labels, test_set, test_labels = train_test
        # select the edges based on the real labels
        edges = gather_edges(big_graph, ways)
        edges_weights = get_subgraph_weight(edges)
        print('ways=', ways, '\t', 'weight=', edges_weights)
        # now, you can do whatever you want... the labels are always between 0 and n_shot-1 like in your code
        # model = train_my_favourite_model(train_set, train_labels)
        # predicted = model.predict_labels(test_set)
        # accuracy = compute_accuracy(predicted, test_labels)
    # and at the end you can compute correlation between weight and accuracy


if __name__ == '__main__':
    # you need the backbone
    # the resnet with .pt is yours
    # the wideresnet with .plk is the one yuqing gave me
    data_paths = {'resnet':'images/latent/miniImagenet/ResNet/layer5/features.pt',
                  'wideresnet':'images/latent/miniImagenet/WideResNet28_10_S2M2_R/last/novel.plk'}
    dot_path = {'resnet':'graphs/resnet18/louvain_dendrogram_communities_1_20.dot',
                'wideresnet':'graphs/wideresnet/louvain_dendrogram_communities_1_20.dot'}
    hard_tasks(data_path=data_paths['wideresnet'], dot_name=dot_path['wideresnet'], n_way=5, n_shot=5, n_val=15)