# xor_pytorch_relu.py
import torch
import torch.nn as nn

#1. 数据准备，XOR 四个点
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]], dtype=torch.float32)
y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]], dtype=torch.float32)

#2. 设备(gpu or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y = X.to(device), y.to(device)

#3. 网络结构，直接用pytorch提供的类的对象，方法来实现
class XORNet(nn.Module):
    def __init__(self, in_dim=2, hidden=4, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.out_act(x)
        return x

model = XORNet().to(device)

# 4) 损失与优化器
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# 5) 训练
model.train()
for epoch in range(2000):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        with torch.no_grad():
            preds = (y_pred >= 0.5).float()
            acc = (preds == y).float().mean().item()
        print(f"Epoch {epoch+1:4d} | loss={loss.item():.6f} | acc={acc*100:.1f}%")

# 6) 验证
model.eval()
with torch.no_grad():
    probs = model(X)
    preds = (probs >= 0.5).float()
    print("\n可能性:\n", probs.cpu().numpy())
    print("标签:\n", preds.cpu().numpy())
