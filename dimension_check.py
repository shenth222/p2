import torch
from transformers import OPTForCausalLM, GPT2Tokenizer, AutoConfig, GenerationConfig
from transformers.models.opt.modeling_opt import OPTAttention
from IPython import embed
import numpy as np
from tqdm import trange
import time
from datasets import load_dataset

MODEL = "../models/opt/opt-125m"
device = "cuda"

config = AutoConfig.from_pretrained(MODEL)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
config.enable_bias = True
config.num_hidden_layers = 1
model = OPTForCausalLM.from_pretrained(MODEL, config=config)


model.to(device)

generation_config = GenerationConfig.from_pretrained(MODEL)
generation_config.max_new_tokens = 1
generation_config.do_sample = False
generation_config.use_cache = True
generation_config.return_dict_in_generate = True
generation_config.output_attentions = True # Tuple(len(new_tokens)), Tuple(len(num_layers)), tensor(bsz, num_heads, tgt_len, src_len)
generation_config.output_hidden_states = True
generation_config.output_scores = True

# test = load_dataset("../../datasets/wikitext", "wikitext-2-raw-v1", split="test")
test = load_dataset("../datasets/c4", split="validation")
test_len = test.num_rows
# test = [{"text":"Today is a sunny day"}]
test_len = 2000
MAX_LENGTH = config.max_position_embeddings
attn_base = []
print("BASE:")
for idx in trange(test_len):
    prompt = (test[idx]["text"])
    model_inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=MAX_LENGTH-1).to(device)
    outputs = model.generate(**model_inputs, generation_config=generation_config)
    attn_base.append(outputs.attentions[0][0][0].half().to("cpu").numpy())

# embed()
def show_outputs(model, model_inputs, outputs):
    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    input_length = model_inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        # | token | token string | log probability | probability
        print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")

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

def compress_QK_per_head(model, config, method):
    tgt_layer_idx = [x for x in range(0,config.num_hidden_layers)]
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

def o_dis(A, B):
    if A.shape != B.shape:
        raise ValueError("Shape mismatch")
    row_num = A.shape[0]
    dis = []
    for idx in range(row_num):
        dis.append(np.linalg.norm(A[idx] - B[idx]))
    return sum(dis) / len(dis)

from scipy.stats import entropy
def kl(base, approximation):
    if base.shape != approximation.shape:
        raise ValueError("Shape mismatch")
    row_num = base.shape[0]
    loss = []
    for idx in range(row_num):
        loss.append(entropy(base[idx][:idx+1], approximation[idx][:idx+1]))
    return sum(loss) / len(loss)

SIMILARITY = {
    "odis": o_dis,
    "kl": kl
}

def similarity(base, out):
    if base.shape[0] != out.shape[0]:
        raise ValueError("Shape mismatch")
    num_heads = base.shape[0]
    sim = []
    for idx in range(num_heads):
        res = {}
        for key in SIMILARITY:
            res[key] = SIMILARITY[key](base[idx], out[idx])
        sim.append(res)
    return sim

import matplotlib.pyplot as plt
COLOR = ['#FF7F50', '#008080', '#FFFF00', '#800080', '#FFA500', '#00FFFF', 
         '#808000', '#FF0000', '#0000FF', '#8B4513', '#A020F0', '#F0E68C']
def plt_sim_res(res, note):
    for key in res:
        sample_nums = len(res[key])
        x = [idx for idx in range(sample_nums)]
        SIM = res[key][0][0].keys()
        head_nums = len(res[key][0])
        for sim in SIM:
            plt.figure(figsize=(8, 6))
            title = f"{key}_{sim}_results"
            for head in range(head_nums):
                y = [sample[head][sim] for sample in res[key]]
                plt.scatter(x, y, marker='.', c=COLOR[head], label=f'head {head}', s=1)
            plt.xlabel('Sample index')
            plt.ylabel('Results')
            plt.title(title)
            plt.grid(False)
            plt.legend(loc='upper right')
            plt.savefig(f"./figure/{note}_{title}.png", dpi=300)
            plt.clf()

import modyfied_opt
out_total = {}
print("TOTAL:")
for key in LOW_RANK_METHOD:
    del model
    model = OPTForCausalLM.from_pretrained(MODEL, config=config)
    start_time = time.time()
    compress_QK_total(model, config, key)
    end_time = time.time()
    model.to(device)
    attn_method = []
    print(f"{key}:")
    for idx in trange(test_len):
        prompt = (test[idx]["text"])
        model_inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=MAX_LENGTH-1).to(device)
        outputs = model.generate(**model_inputs, generation_config=generation_config)
        attn_method.append(outputs.attentions[0][0][0].half().to("cpu").numpy())
    out_total[key] = attn_method
    # print(f"Method: {key} takes {end_time - start_time}s")
sim_res_total = {} # sim_res_total[compress_method][sample_index][head_idx][sim_method]
for key in out_total:
    res = []
    for idx in trange(test_len):
        res.append(similarity(attn_base[idx], out_total[key][idx]))
    sim_res_total[key] = res
del out_total
plt_sim_res(sim_res_total, "Total")
del sim_res_total

out_per_head = {}
print("PER_HEAD:")
for key in LOW_RANK_METHOD:
    del model
    model = OPTForCausalLM.from_pretrained(MODEL, config=config)
    start_time = time.time()
    compress_QK_per_head(model, config, key)
    end_time = time.time()
    model.to(device)
    attn_method = []
    print(f"{key}:")
    for idx in trange(test_len):
        prompt = (test[idx]["text"])
        model_inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=MAX_LENGTH-1).to(device)
        outputs = model.generate(**model_inputs, generation_config=generation_config)
        attn_method.append(outputs.attentions[0][0][0].half().to("cpu").numpy())
    out_per_head[key] = attn_method
    # print(f"Method: {key} takes {end_time - start_time}s")
sim_res_per_head = {} # sim_res_per_head[compress_method][sample_index][head_idx][sim_method]
for key in out_per_head:
    res = []
    for idx in trange(test_len):
        res.append(similarity(attn_base[idx], out_per_head[key][idx]))
    sim_res_per_head[key] = res

plt_sim_res(sim_res_per_head, "Per_head")
