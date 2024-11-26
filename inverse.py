import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# 定义设备（CPU或GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义模型
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyModel, self).__init__()
        # 定义模型的层
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self):
        # 定义前向传播过程
        # x = self.linear.weight # A_T
        # a = self.linear(x) # A_T @ A
        # b = torch.inverse(a)
        # c = torch.mm(x.t(), b)
        # out = torch.mm(c, x)
        x = self.linear.weight.t()
        a = torch.pinverse(x)
        out = torch.mm(x, a)
        return out

seed = 1234
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

model = MyModel(768, 384).to(device)
input = model.linear.weight.detach()
# 定义损失函数和优化器
def criterion(matrix1, matrix2):
    diff = matrix1 - matrix2
    frobenius_norm = torch.norm(diff, p='fro')
    return frobenius_norm

optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

label = torch.eye(768)
num_epochs = 1000
# 训练循环
for epoch in range(num_epochs):
    label = label.to(device)

    # 前向传播
    outputs = model()
    loss = criterion(outputs, label)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())

    # 打印训练过程信息（可选）
    print('Epoch [{}/{}], Lr: {}, Loss: {}'.format(epoch+1, num_epochs, optimizer.state_dict()['param_groups'][0]['lr'], loss.item()))

from IPython import embed
# embed()