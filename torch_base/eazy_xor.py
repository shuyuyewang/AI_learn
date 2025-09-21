# xor_pytorch_stable_relu.py
import torch
import torch.nn as nn

# 1) 固定随机种子，确保可复现
torch.manual_seed(0)

# 2) 数据
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]], dtype=torch.float32)
y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]], dtype=torch.float32)

# 3) 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y = X.to(device), y.to(device)

# 4) 模型：ReLU 中间层（不在 forward 里做 Sigmoid，直接输出 logits）
class XORNet(nn.Module):
    def __init__(self, in_dim=2, hidden=8, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

        # ——关键：初始化更友好——
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.act(self.fc1(x))
        logits = self.fc2(x)        # 直接输出 logits
        return logits

model = XORNet().to(device)

# 5) 损失与优化器：用 BCEWithLogitsLoss（内部自带 Sigmoid，数值稳定）
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 稍微小一点

# 6) 训练
model.train()
for epoch in range(3000):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 300 == 0:
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == y).float().mean().item()
        print(f"Epoch {epoch+1:4d} | loss={loss.item():.6f} | acc={acc*100:.1f}%")

# 7) 验证
model.eval()
with torch.no_grad():
    logits = model(X)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    print("\n可能性:\n", probs.cpu().numpy())
    print("标签:\n", preds.cpu().numpy())
