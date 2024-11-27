import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from transformers import AutoModel, GPT2Tokenizer, AutoConfig, GenerationConfig

# 定义设备（CPU或GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL = "../../models/llama/llama2/llama-2-7b-hf"
config = AutoConfig.from_pretrained(MODEL)
input_model = AutoModel.from_pretrained(MODEL, config=config)
W_Q = []
for layer_idx in range(config.num_hidden_layers):
    W_Q.append(input_model.layers[layer_idx].self_attn.q_proj.weight.detach())
del input_model

class TransModel(nn.Module):
    def __init__(self):
        super(TransModel, self).__init__()

        self.linear1 = nn.Linear(4096, 4096*4)
        self.linear2 = nn.Linear(4096*4, 2048)

    def forward(self, x):
        return self.linear2(self.linear1(x))

def criterion(A, I):
    # torch.mm(torch.mm(A, torch.inverse(torch.mm(A.t(), A))), A.t())
    diff = torch.mm(A, torch.linalg.pinv(A)) - I
    return torch.norm(diff, p='fro')

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

import torch_directml
dml = torch_directml.device()
model = TransModel().to(dml)
label = torch.eye(4096).to(dml)
num_epochs = 10
optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

for epoch in range(num_epochs):
    label = label.to(dml)
    for iteration, input in enumerate(W_Q):
        output = model(input.to(dml))
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        
        print('Epoch [{}/{}], Iter [{}/32] Lr: {}, Loss: {}'.format(epoch+1, num_epochs, iteration+1, optimizer.state_dict()['param_groups'][0]['lr'], loss.item()))



# 定义模型
# class MyModel(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(MyModel, self).__init__()
#         # 定义模型的层
#         self.linear = nn.Linear(input_size, hidden_size)

#     def forward(self):
#         # 定义前向传播过程
#         # x = self.linear.weight # A_T
#         # a = self.linear(x) # A_T @ A
#         # b = torch.inverse(a)
#         # c = torch.mm(x.t(), b)
#         # out = torch.mm(c, x)
#         x = self.linear.weight.t()
#         a = torch.pinverse(x)
#         out = torch.mm(x, a)
#         return out

# seed = 1234
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# model = MyModel(768, 384).to(device)
# input = model.linear.weight.detach()
# # 定义损失函数和优化器
# def criterion(matrix1, matrix2):
#     diff = matrix1 - matrix2
#     frobenius_norm = torch.norm(diff, p='fro')
#     return frobenius_norm

# optimizer = optim.Adam(model.parameters(), lr=0.1)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# label = torch.eye(768)
# num_epochs = 1000
# # 训练循环

# from IPython import embed
# # embed()

