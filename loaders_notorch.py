import functools
import pickle
import numpy as np


def get_labels_stats(labels):
    labels_set = list(set(labels.tolist()))
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
        subset_labels.append(np.array([new_l]*n_sample_per_label, dtype=np.int64))
    subset_data = np.concatenate(subset_data, axis=0)
    subset_labels = np.concatenate(subset_labels, axis=0)
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
    train_set = np.concatenate(train_set, axis=0)
    test_set = np.concatenate(test_set, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    return train_set, train_labels, test_set, test_labels

def load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        labels = [np.full(shape=len(data[key]), fill_value=key) for key in data]
        data = [features for key in data for features in data[key]]
        dataset = dict()
        dataset['data'] = np.stack(data, axis=0)
        dataset['labels'] = np.concatenate(labels)
        return dataset

@functools.lru_cache()
def get_dataset_from_datapath(data_path):
    original_data = []
    original_labels = []
    paths = data_path.split('&')
    for path in paths:
        if data_path.endswith('.plk') or data_path.endswith('.pkl'):
            dataset = load_pickle(path)
        else:
            assert False
        original_data.append(dataset['data'])
        original_labels.append(dataset['labels'])
    original_data = np.concatenate(original_data)
    original_labels = np.concatenate(original_labels)
    return original_data, original_labels

def get_train_test_datasets(data_path, n_way, n_shot, n_val):
    original_data, original_labels = get_dataset_from_datapath(data_path)
    data, labels = data_subset(original_data, original_labels, n_way)
    return split_train_test(data, labels, n_shot, n_val)

def get_train_test_datasets_labels(data_path, n_way, n_shot, n_val, ways=None):
    original_data, original_labels = get_dataset_from_datapath(data_path)
    num_labels, n_sample_per_label = get_labels_stats(original_labels)
    if ways is None:
        ways = np.random.choice(range(num_labels), n_way, replace=False)
    selected_labels = [int(original_labels[i*n_sample_per_label]) for i in ways]
    data, labels = data_subset(original_data, original_labels, n_way, ways=ways)
    return split_train_test(data, labels, n_shot, n_val), selected_labels

def get_all_pairs_datasets(data_path, n_way, crop, parts=None):
    assert n_way == 2
    original_data, original_labels = get_dataset_from_datapath(data_path)
    num_labels, n_sample_per_label = get_labels_stats(original_labels)
    if parts is None:
        start_i = 0
        yield int(num_labels) * int(num_labels-1) // 2
    else:
        start_i = parts
        yield parts * (num_labels - parts)
    for i in range(start_i, num_labels):
        if crop and i == 5:
            break
        end_j = parts if parts is not None else i
        for j in range(0, end_j):
            ways = np.array([i, j])
            label_a = int(original_labels[i*n_sample_per_label])
            label_b = int(original_labels[j*n_sample_per_label])
            data, labels = data_subset(original_data, original_labels, n_way, ways)
            yield data, labels, label_a, label_b, 0, 1
