import torch
from transformers import OPTForCausalLM, GPT2Tokenizer, AutoConfig, GenerationConfig
from transformers.models.opt.modeling_opt import OPTAttention
from IPython import embed
import numpy as np
from tqdm import trange
import time
from datasets import load_dataset

MODEL = "../../models/opt/opt-125m"
device = "cpu"

config = AutoConfig.from_pretrained(MODEL)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
config.enable_bias = True
config.num_hidden_layers = 1

# test = load_dataset("../../datasets/wikitext", "wikitext-2-raw-v1", split="test")
test = load_dataset("../../datasets/c4", split="validation")
test_len = test.num_rows
# test = [{"text":"Today is a sunny day"}]
# test_len = 2000
max_length = config.max_position_embeddings
stride = 1024

def ppl(model):
    total_ppl = 0
    for idx in trange(test_len):
        prompt = (test[idx]["text"])
        encodings = tokenizer([prompt], return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            # Accumulate the total negative log-likelihood and the total number of tokens
            num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
            nll_sum += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
        ppl = torch.exp(avg_nll)
        total_ppl += ppl
    return total_ppl

from packages import LOW_RANK_METHOD, compress_QK_total, compress_QK_per_head
model = None
for key in LOW_RANK_METHOD:
    del model
    model = OPTForCausalLM.from_pretrained(MODEL, config=config)
    compress_QK_total(model, config, key)
    model.to(device)
    res = ppl(model)
    print(f"{key}_total: {res}")

for key in LOW_RANK_METHOD:
    del model
    model = OPTForCausalLM.from_pretrained(MODEL, config=config)
    model = compress_QK_per_head(model, config, key)
    model.to(device)
    res = ppl(model)
    print(f"{key}_per_head: {res}")


