# ==============================
# minimal_bp_xor_clean.py
# 分区：数据集 | 模型 | 训练 | 测试 | 可视化
# ==============================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ========== [A] 数据集区域 (Dataset Region) ==========
# 1) 训练数据（XOR 四点）
train_X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.],
], dtype=torch.float32)                                    # (4,2)

train_y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.],
], dtype=torch.float32)                                    # (4,1)

# 2) 额外测试数据（可按需增删）
test_X = torch.tensor([
    [0.2, 0.2],
    [0.8, 0.2],
    [0.5, 0.5],
    [1.2, 0.9]
], dtype=torch.float32)                                    # (M,2)

# 3) 可视化网格范围（决策边界背景）
VIS_X_MIN, VIS_X_MAX = -0.5, 1.5
VIS_Y_MIN, VIS_Y_MAX = -0.5, 1.5
VIS_GRID_N = 300  # 网格密度（越大越细）

# ========== [B] 模型区域 (Model Region) ==========
# 2 -> 8 -> 1，隐藏层 ReLU；输出为 logit（不给 sigmoid）
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# 损失与优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)  # 小数据可用较大学习率

# ========== [C] 训练区域 (Training Region) ==========
EPOCHS = 2000
log_every = 200
loss_history = []

for step in range(1, EPOCHS + 1):
    logits = model(train_X)                 # 前向
    loss = criterion(logits, train_y)       # 损失

    optimizer.zero_grad()
    loss.backward()                         # 反向
    optimizer.step()                        # 更新

    loss_history.append(loss.item())
    if step % log_every == 0:
        with torch.no_grad():
            probs_eval = torch.sigmoid(model(train_X))
            preds_eval = (probs_eval > 0.5).float()
            acc = (preds_eval.eq(train_y)).float().mean().item()
        print(f"[Train] step {step:4d} | loss {loss.item():.4f} | acc {acc:.2f}")

# ========== [D] 测试区域 (Testing Region) ==========
with torch.no_grad():
    # 1) 训练集上的最终表现
    train_probs = torch.sigmoid(model(train_X))
    train_preds = (train_probs > 0.5).float()

    print("\n=== 训练集 (XOR 四点) 最终预测 ===")
    for xi, yi, pi in zip(train_X, train_y, train_probs):
        print(f"Input {xi.tolist()} | True {int(yi.item())} "
              f"| Pred_prob {pi.item():.3f} | Pred_label {int(pi.item()>0.5)}")

    # 2) 额外测试点
    test_probs = torch.sigmoid(model(test_X))
    test_preds = (test_probs > 0.5).float()

    print("\n=== 额外测试点的预测 ===")
    for xi, pi in zip(test_X, test_probs):
        print(f"Input {xi.tolist()} | Pred_prob {pi.item():.3f} "
              f"| Pred_label {int(pi.item()>0.5)}")

# ========== [E] 可视化区域 (Visualization Region) ==========
xx, yy = np.meshgrid(
    np.linspace(VIS_X_MIN, VIS_X_MAX, VIS_GRID_N),
    np.linspace(VIS_Y_MIN, VIS_Y_MAX, VIS_GRID_N),
)
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

with torch.no_grad():
    zz = torch.sigmoid(model(grid)).reshape(xx.shape).numpy()

plt.figure(figsize=(5.5, 5.5))
# 背景概率
plt.contourf(xx, yy, zz, levels=50, alpha=0.5)
# 决策边界线 (p=0.5)
plt.contour(xx, yy, zz, levels=[0.5], linewidths=2, colors='black')

# 训练集正确/错误标注
with torch.no_grad():
    train_probs_plot = torch.sigmoid(model(train_X))
    train_preds_plot = (train_probs_plot > 0.5).float()
correct_mask = (train_preds_plot.squeeze() == train_y.squeeze())
wrong_mask   = ~correct_mask

plt.scatter(train_X[correct_mask, 0], train_X[correct_mask, 1],
            c='green', s=120, marker='o', edgecolors='k', label='Train Correct')
plt.scatter(train_X[wrong_mask, 0], train_X[wrong_mask, 1],
            c='red', s=150, marker='x', linewidths=3, label='Train Wrong')

# 测试点（蓝色方块 + 标注）
with torch.no_grad():
    test_probs_plot = torch.sigmoid(model(test_X))
    test_preds_plot = (test_probs_plot > 0.5).float()

plt.scatter(test_X[:, 0], test_X[:, 1],
            c='blue', s=90, marker='s', edgecolors='k', label='Test Points')

# 在每个测试点旁边加预测结果
for (x, y_val), prob, pred in zip(test_X.numpy(), test_probs_plot.numpy(), test_preds_plot.numpy()):
    plt.text(x+0.05, y_val+0.05, f"{int(pred[0])} ({prob[0]:.2f})",
             color="blue", fontsize=9, weight="bold")

plt.title("XOR Decision Boundary / Train & Test Points")
plt.xlabel("x1"); plt.ylabel("x2")
plt.xlim(VIS_X_MIN, VIS_X_MAX); plt.ylim(VIS_Y_MIN, VIS_Y_MAX)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
