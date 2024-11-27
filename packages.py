import torch
import numpy as np

def SVD_approximation(A, ratios):
    # the dimension of column is same as before
    if not isinstance(A, np.ndarray):
        A = A.numpy()
    U, sigma, VT = np.linalg.svd(A)
    r = int(len(sigma)*ratios)
    sigma_r = np.diag(sigma[:r])
    A_r = U[:, :r] @ sigma_r @ VT[:r, :]
    return A_r

from sklearn.decomposition import PCA
def pca(data, ratios=0.5):
    pca = PCA(n_components=int(data.shape[-1]*ratios))
    pca.fit(data)
    new_data = pca.transform(data)
    return new_data

def random_projection(X, ratios=0.5):
    n_features = X.shape[-1]
    np.random.seed(42)
    random_matrix = np.random.randn(int(n_features*ratios), n_features)
    X_new = np.dot(X, random_matrix.T)
    return X_new

from sklearn.decomposition import FastICA
def ica(X, ratios=0.5):
    ica = FastICA(n_components=int(X.shape[-1]*ratios))
    S = ica.fit_transform(X)
    return S

from sklearn.manifold import MDS
def mds_dimensionality_reduction(data, ratios=0.5):
    mds = MDS(n_components=int(data.shape[-1]*ratios), dissimilarity='euclidean')
    embedding = mds.fit_transform(data)
    return embedding

from sklearn.feature_selection import VarianceThreshold
def variance_threshold(X, threshold=0.005):
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    return X_filtered

from sklearn.manifold import TSNE
def tsne(X, ratios=0.5):
    tsne = TSNE(n_components=int(X.shape[-1]*ratios), random_state=0)
    X_embedded = tsne.fit_transform(X)
    return X_embedded

LOW_RANK_METHOD = {
    "PCA": pca,
    # "RP": random_projection,
    # "ICA": ica,
    # "MDS": mds_dimensionality_reduction,
    # "vt": variance_threshold,
    # "tsne": tsne
}

from torch import nn
def compress_QK_total(model, config, method):
    tgt_layer_idx = [x for x in range(0,config.num_hidden_layers)]
    for idx in range(config.num_hidden_layers):
        if idx not in tgt_layer_idx: continue
        q_weight = model.model.decoder.layers[idx].self_attn.q_proj.weight.detach().numpy().T
        if config.enable_bias is True:
            q_bias = model.model.decoder.layers[idx].self_attn.q_proj.bias.detach().numpy()
            q_weight = np.vstack((q_weight, q_bias))
        new_q = LOW_RANK_METHOD[method](q_weight)
        if config.enable_bias is True:
            q_weight = new_q[:-1, :].T
            q_bias = new_q[-1, :]
        else:
            q_weight = new_q.T
        q_linear = nn.Linear(in_features=q_weight.shape[1], out_features=q_weight.shape[0], bias=config.enable_bias)
        q_linear.weight = nn.Parameter(torch.tensor(q_weight, dtype=torch.float32))
        if config.enable_bias is True:
            q_linear.bias = nn.Parameter(torch.tensor(q_bias, dtype=torch.float32))
        model.model.decoder.layers[idx].self_attn.q_proj = q_linear
        ##################################################################################################################
        k_weight = model.model.decoder.layers[idx].self_attn.k_proj.weight.detach().numpy().T
        if config.enable_bias is True:
            k_bias = model.model.decoder.layers[idx].self_attn.k_proj.bias.detach().numpy()
            k_weight = np.vstack((k_weight, k_bias))
        new_k = LOW_RANK_METHOD[method](k_weight)
        if config.enable_bias is True:
            k_weight = new_k[:-1, :].T
            k_bias = new_k[-1, :]
        else:
            k_weight = new_k.T
        k_linear = nn.Linear(in_features=k_weight.shape[1], out_features=k_weight.shape[0], bias=config.enable_bias)
        k_linear.weight = nn.Parameter(torch.tensor(k_weight, dtype=torch.float32))
        if config.enable_bias is True:
            k_linear.bias = nn.Parameter(torch.tensor(k_bias, dtype=torch.float32))
        model.model.decoder.layers[idx].self_attn.k_proj = k_linear
    return model

def compress_QK_per_head(model, config, method):
    tgt_layer_idx = [x for x in range(3,9)]
    for idx in range(config.num_hidden_layers):
        if idx not in tgt_layer_idx: continue
        q_weight = model.model.decoder.layers[idx].self_attn.q_proj.weight.detach().numpy().T
        if config.enable_bias is True:
            q_bias = model.model.decoder.layers[idx].self_attn.q_proj.bias.detach().numpy()
            q_weight = np.vstack((q_weight, q_bias))
        q_split = np.hsplit(q_weight, config.num_attention_heads)
        new_q = []
        for q in q_split:
            new_q.append(LOW_RANK_METHOD[method](q))
        new_q = np.hstack(new_q)
        if config.enable_bias is True:
            q_weight = new_q[:-1, :].T
            q_bias = new_q[-1, :]
        else:
            q_weight = new_q.T
        q_linear = nn.Linear(in_features=q_weight.shape[1], out_features=q_weight.shape[0], bias=config.enable_bias)
        q_linear.weight = nn.Parameter(torch.tensor(q_weight, dtype=torch.float32))
        if config.enable_bias is True:
            q_linear.bias = nn.Parameter(torch.tensor(q_bias, dtype=torch.float32))
        model.model.decoder.layers[idx].self_attn.q_proj = q_linear
        ######################################################################################
        k_weight = model.model.decoder.layers[idx].self_attn.k_proj.weight.detach().numpy().T
        if config.enable_bias is True:
            k_bias = model.model.decoder.layers[idx].self_attn.k_proj.bias.detach().numpy()
            k_weight = np.vstack((k_weight, k_bias))
        k_split = np.hsplit(k_weight, config.num_attention_heads)
        new_k = []
        for k in k_split:
            new_k.append(LOW_RANK_METHOD[method](k))
        new_k = np.hstack(new_k)
        if config.enable_bias is True:
            k_weight = new_k[:-1, :].T
            k_bias = new_k[-1, :]
        else:
            k_weight = new_k.T
        k_linear = nn.Linear(in_features=k_weight.shape[1], out_features=k_weight.shape[0], bias=config.enable_bias)
        k_linear.weight = nn.Parameter(torch.tensor(k_weight, dtype=torch.float32))
        if config.enable_bias is True:
            k_linear.bias = nn.Parameter(torch.tensor(k_bias, dtype=torch.float32))
        model.model.decoder.layers[idx].self_attn.k_proj = k_linear
    return model


def compress_QK_w_inverse_per_head(model, config):
    tgt_layer_idx = [x for x in range(0,config.num_hidden_layers)]
    for idx in range(config.num_hidden_layers):
        if idx not in tgt_layer_idx: continue
        q_weight = model.model.decoder.layers[idx].self_attn.q_proj.weight.detach().numpy().T
        q_split = np.hsplit(q_weight, config.num_attention_heads)
        new_q = []
        for q in q_split:
            # new_q.append(LOW_RANK_METHOD[method](q))
            pass

