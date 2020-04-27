import argparse
import collections as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from tqdm import tqdm
from utils import GridSearch
from loaders import get_train_test_datasets, get_dataset_from_datapath
from classifiers import features_classification


class X_To_Z(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(X_To_Z, self).__init__()
        self.kernel = nn.Linear(input_dim, output_dim)
        self.non_linearity = nn.ReLU()

    def forward(self, x):
        x = self.kernel(x)
        x = self.non_linearity(x)
        return x

class Z_To_Y(nn.Module):

    def __init__(self, input_dim, num_Y_classes):
        super(Z_To_Y, self).__init__()
        self.kernel = nn.Linear(input_dim, num_Y_classes)

    def forward(self, x):
        return self.kernel(x)


class MI_Kernel(nn.Module):
    
    def __init__(self, input_dim, z_dim):
        super(MI_Kernel, self).__init__()
        self.kernel = nn.Parameter(data=torch.zeros(size=(z_dim, input_dim), 
                                                    dtype=torch.float32),
                                   requires_grad=True)

    def forward(self, x, z):
        # k batch of positive examples
        # l batch of negative examples
        # i,j kernel
        return torch.einsum('ki,ij,lj->kl', z, self.kernel, x)


def loopy(items):
    while True:
        for item in iter(items):
            yield item

def get_transfered_dataset(x_support, x_train, x_test, params):
    x_transfered = torch.cat([x_train, x_test, x_support], dim=0).to(params.device)
    loader = DataLoader(x_transfered, batch_size=params.batch_size,
                                      sampler=RandomSampler(x_transfered, replacement=True),
                                      drop_last=True)
    return loopy(loader)

def get_support_dataset(x_support, y_support, params):
    x_support, y_support = x_support.to(params.device), y_support.to(params.device)
    dataset = TensorDataset(x_support, y_support)
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, drop_last=False)
    return loopy(loader)

def update_train_progress(progress, infos, step, sl, el, mil):
    alpha = 0.99
    if step == 0:
        infos['sl'] = float(sl.item())
        infos['el'] = float(el.item())
        infos['mil'] = float(mil.item())
    infos['sl'] = alpha * infos['sl'] + (1 - alpha) * float(sl.item())
    infos['el'] = alpha * infos['el'] + (1 - alpha) * float(el.item())
    infos['mil'] = alpha * infos['mil'] + (1 - alpha) * float(mil.item())
    desc = 'LS=%.5f EL=%.5f MIL=%.5f'%(infos['sl'], infos['el'], infos['mil'])
    progress.set_description(desc=desc)
    progress.update()

def entropy_loss(y):
    p_log_p = F.softmax(y, dim=1) * F.log_softmax(y, dim=1)
    loss = -1.0 * p_log_p.sum(dim=1).mean()
    return loss

def info_nce_loss(mi_kernel, x, z):
    try:
        labels = info_nce_loss.labels
    except AttributeError:
        info_nce_loss.labels = torch.LongTensor(list(range(int(z.shape[0]))))
        if z.is_cuda:
            info_nce_loss.labels = info_nce_loss.labels.to(z.get_device())
        labels = info_nce_loss.labels
    logits = mi_kernel(x, z)
    return nn.functional.cross_entropy(logits, labels)

def info_jsd_loss(mi_kernel, x, z):
    try:
        targets = info_jsd_loss.targets
    except AttributeError:
        info_jsd_loss.targets = 1 - 2*torch.eye(n=int(x.shape[0]))
        if z.is_cuda:
            info_jsd_loss.targets = info_jsd_loss.targets.to(z.get_device())
        targets = info_jsd_loss.targets
    logits = mi_kernel(x, z)
    pre_softplus = targets * logits
    critic = F.softplus(pre_softplus, threshold=20)
    critic_fake = critic - torch.diag(critic)
    critic_real = torch.diag(critic)
    loss = torch.mean(critic_fake) + torch.mean(critic_real)
    return loss

def get_mi_loss(mi_loss):
    if mi_loss == 'NCE':
        return info_nce_loss
    if mi_loss == 'JSD':
        return info_jsd_loss
    assert False

def learn_x_to_z(x_support, y_support, x_train, x_test, params):
    input_dim, output_dim = int(x_train.shape[1]), int(torch.max(y_support)+1)
    transfered_dataset = get_transfered_dataset(x_support, x_train, x_test, params)
    support_dataset = get_support_dataset(x_support, y_support, params)
    x_to_z = X_To_Z(input_dim, params.z_dim)
    z_to_y = Z_To_Y(params.z_dim, output_dim)
    mi_kernel = MI_Kernel(input_dim, params.z_dim)
    mi_kernel.train()
    mi_kernel.to(params.device)
    mi_loss = get_mi_loss(params.mi_loss)
    model = nn.Sequential(x_to_z, z_to_y)
    model.train()
    model.to(params.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters())+list(mi_kernel.parameters()))
    progress = tqdm(total=params.n_steps, leave=False, desc='X->Z->Y')
    infos = dict()
    for items in zip(range(params.n_steps), support_dataset, transfered_dataset):
        step, (x, labels), x_t = items
        optimizer.zero_grad()
        sl = nn.functional.cross_entropy(model(x), labels)  # learn classifier over support
        z_t = x_to_z(x_t)
        y_t = z_to_y(z_t)
        el_t = entropy_loss(y_t)  # minimize entropy over new distribution to reduce confusion
        mil_t = mi_loss(mi_kernel, x_t, z_t)  # preserve mutual information
        loss = params.lbda_sl*sl + params.lbda_el*el_t + params.lbda_mil*mil_t
        loss.backward()
        optimizer.step()
        update_train_progress(progress, infos, step, sl, el_t, mil_t)
    progress.close()
    x_to_z.to('cpu')
    x_to_z.eval()
    return x_to_z

def learn_z_to_y_bar(infos, x_to_z, x_train, x_test, y_train, y_test, params):
    with torch.no_grad():
        z_train = x_to_z(x_train)
        z_test = x_to_z(x_test)
    accs = features_classification(z_train, y_train, z_test, y_test,
                                   params.n_way, params.classifier, params.normalization, params)
    infos['train_acc'].append(accs[0])
    infos['test_acc'].append(accs[1])

def run_test(infos, x_support, y_support, params):
    datasets = get_train_test_datasets(params.novel_datapath, params.n_way, params.n_shot, params.n_val)
    x_train, y_train, x_test, y_test = datasets
    x_to_z = learn_x_to_z(x_support, y_support, x_train, x_test, params)
    learn_z_to_y_bar(infos, x_to_z, x_train, x_test, y_train, y_test, params)

def update_sweep_progress(progress, infos):
    train_acc, test_acc = float(np.mean(infos['train_acc'])), float(np.mean(infos['test_acc']))
    desc = 'train_acc=%.2f%% test_acc=%.2f%%'%(train_acc, test_acc)
    progress.set_description(desc=desc)
    progress.update()

def remap_labels(y_support):
    y_support = y_support.cpu().numpy().tolist()
    labels = list(set(y_support))
    new_labels = dict(zip(labels, range(len(labels))))
    y_support = [new_labels[label] for label in y_support]
    y_support = torch.LongTensor(y_support)
    return y_support

def sweep_grid_search(grid_search):
    x_support, y_support = get_dataset_from_datapath(grid_search.get('train_datapath'), normalization='l2')
    y_support = remap_labels(y_support)
    print(grid_search.get_constant_keys())
    print('')
    for params in grid_search.get_params():
        print(grid_search.get_variable_keys(params))
        progress = tqdm(total=params.num_tests, leave=True, desc='Number of tests')
        infos = ct.defaultdict(list)
        for _ in range(params.num_tests):
            run_test(infos, x_support, y_support, params)
            update_sweep_progress(progress, infos)
        progress.close()
        print('')


def get_grid_search(num_tests, device):
    grid_search = GridSearch()
    grid_search.add_range('z_dim', [320])
    grid_search.add_range('lbda_sl', [1.])
    grid_search.add_range('lbda_el', [0.])
    grid_search.add_range('lbda_mil', [1.])
    grid_search.add_range('mi_loss', ['JSD', 'NCE'])
    grid_search.add_range('n_steps', [2000])
    grid_search.add_range('batch_size', [12])
    grid_search.add_range('n_way', [5])
    grid_search.add_range('n_shot', [5])
    grid_search.add_range('n_val', [100])
    grid_search.add_range('classifier', ['logistic_regression'])
    grid_search.add_range('normalization', ['l2'])
    grid_search.add_range('novel_datapath', ['images/WideResNet28_10_S2M2_R/last/novel.plk'])
    grid_search.add_range('train_datapath', ['images/WideResNet28_10_S2M2_R/last/train.plk'])
    grid_search.add_range('num_tests', [num_tests])
    grid_search.add_range('device', [device])
    return grid_search


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Few label learning.')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    sanity_check_parser = parser.add_mutually_exclusive_group(required=True)
    sanity_check_parser.add_argument('--sanity_check', dest='num_tests', action='store_const', const=2)
    sanity_check_parser.add_argument('--brief_overview', dest='num_tests', action='store_const', const=100)
    sanity_check_parser.add_argument('--full_training', dest='num_tests', action='store_const', const=1000)
    parser.set_defaults(num_tests=1000)
    args = parser.parse_args()
    grid_search = get_grid_search(args.num_tests, args.device)
    sweep_grid_search(grid_search)
