import itertools
import torch
from tqdm import trange
import numpy as np
from torch.optim import Adam
from models import get_device
import torch.nn as nn

class PiModel(nn.Module):

    def __init__(self):
        super(PiModel, self).__init__()

    def init_epoch(self, n_steps):
        pass

    def update_step(self, step):
        pass

    def init_inference(self):
        pass

    def get_opt(self):
        return Adam(self.parameters())

    def get_device(self):
        return next(self.parameters()).device

    def regularization_loss(self):
        device = self.get_device()
        return torch.zeros((1,), dtype=torch.float32).to(device)

    def forward(self, x):
        return x


def embed_into_graph(model, train_set):
    train_set= train_set.to(model.get_device())
    with torch.no_grad():
        z_latent = model(train_set)
        return z_latent

def get_bad_edges(labels):
    batch_size = int(labels.shape[0])
    bad_edges = []
    for a in range(batch_size):
        for b in range(a):
            if labels[a] != labels[b]:
                bad_edges.append((a, b))
    return bad_edges

def get_good_edges(labels):
    batch_size = int(labels.shape[0])
    good_edges = []
    for a in range(batch_size):
        for b in range(a):
            if labels[a] == labels[b]:
                good_edges.append((a, b))
    return good_edges

def get_inter_intra_edges(labels):
    bad_edges = get_bad_edges(labels)
    good_edges = get_good_edges(labels)
    return bad_edges, good_edges

def compute_smoothness(z_latent, labels, loss_fn, edge_selector):
    edges_selected = edge_selector(labels)
    loss = loss_fn(z_latent, edges_selected)
    return loss

def train_graph_embedder(train_set, train_labels, model, loss_fn, edge_selector, n_steps):
    model.init_epoch(n_steps)
    model.train()
    device = 'cpu'
    optimizer = model.get_opt()
    model.to(device)
    batch = train_set.to(device)
    edges_selected = edge_selector(train_labels)
    progress = trange(n_steps, desc='Batch', leave=False)
    for step in progress:
        optimizer.zero_grad()
        model.update_step(step)
        z_latent = model(batch)
        smoothness_loss = loss_fn(z_latent, edges_selected)
        regularization_loss = model.regularization_loss()
        loss = smoothness_loss + regularization_loss
        loss.backward()
        optimizer.step()
        x_avg = torch.mean(torch.norm(batch, dim=1)).cpu().item()
        z_avg = torch.mean(torch.norm(z_latent, dim=1)).cpu().item()
        infos = float(loss.cpu().item()), float(smoothness_loss.cpu().item())
        infos = infos + (float(regularization_loss.cpu().item()), x_avg, z_avg)
        progress.set_description('Loss=%.4f Smooth=%.5f Regul=%.4f x_avg=%.2f z_avg=%.2f'%infos)
        progress.refresh()
    model.init_inference()
    model.eval()
    return model

