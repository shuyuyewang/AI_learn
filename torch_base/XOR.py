# xor_line_band_balanced_testonly.py
# 满足：
# 1) 训练集：50% 为 1（|x-y|<eps），50% 为 0（|x-y|>=eps），标签由代码判定
# 2) 可视化：背景 0→红、1→绿 概率渐变，仅显示测试集 50 个点（均匀分布采样）
#    预测=真值 → 紫色圆；预测≠真值 → 黑色圆
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ========== [0] 设备 & 工具 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
to_cpu = lambda t: t.detach().cpu()

# 可视化与采样范围
VIS_X_MIN, VIS_X_MAX = -0.5, 1.5
VIS_Y_MIN, VIS_Y_MAX = -0.5, 1.5

# ========== [1] 数据集生成 ==========
def make_train_on_line_balanced(n_total=4000, eps=0.03, seed=0):
    """
    训练集：一半为正类(1)：|x - y| < eps（在 y=x 带宽内）
            一半为负类(0)：|x - y| >= eps（带宽外）
    采样范围使用全局 VIS_*。
    """
    assert n_total % 2 == 0, "n_total 必须是偶数以保证 50/50"
    rng = np.random.default_rng(seed)
    n_pos = n_total // 2
    n_neg = n_total // 2

    # 正类：在直线附近（沿正交方向偏移到 (-eps, eps)）
    u = rng.uniform(VIS_X_MIN, VIS_X_MAX, size=(n_pos, 1)).astype(np.float32)
    d = rng.uniform(-eps * 0.98, eps * 0.98, size=(n_pos, 1)).astype(np.float32)
    X_pos = np.hstack([u, u + d]).astype(np.float32)

    # 负类：均匀采样后排除带宽内的点
    X_neg = []
    need = n_neg
    while need > 0:
        cand = rng.uniform(
            [VIS_X_MIN, VIS_Y_MIN], [VIS_X_MAX, VIS_Y_MAX],
            size=(need * 2, 2)
        ).astype(np.float32)  # 多采一些以减少循环次数
        mask = np.abs(cand[:, 0] - cand[:, 1]) >= eps
        picked = cand[mask]
        if picked.shape[0] > need:
            picked = picked[:need]
        X_neg.append(picked)
        need -= picked.shape[0]
    X_neg = np.concatenate(X_neg, axis=0).astype(np.float32)

    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.vstack([
        np.ones((n_pos, 1), dtype=np.float32),
        np.zeros((n_neg, 1), dtype=np.float32)
    ])

    # 打乱
    idx = rng.permutation(X.shape[0])
    X, y = X[idx], y[idx]
    return torch.from_numpy(X), torch.from_numpy(y)

def latin_hypercube_2d(n, seed=0):
    """
    在矩形区域做 2D Latin Hypercube 采样，尽量均匀分布。
    返回 shape=(n,2) 的点。
    """
    rng = np.random.default_rng(seed)
    xs = (np.arange(n) + rng.random(n)) / n
    ys = (np.arange(n) + rng.random(n)) / n
    rng.shuffle(xs)
    rng.shuffle(ys)
    X = VIS_X_MIN + xs * (VIS_X_MAX - VIS_X_MIN)
    Y = VIS_Y_MIN + ys * (VIS_Y_MAX - VIS_Y_MIN)
    pts = np.stack([X, Y], axis=1).astype(np.float32)
    return pts

def label_on_line(X_np, eps=0.03):
    """根据 |x-y|<eps 生成标签：1=在 y=x 带宽内，0=否则。"""
    return (np.abs(X_np[:, 0] - X_np[:, 1]) < eps).astype(np.float32).reshape(-1, 1)

def prepare_data(train_total=4000, test_n=50, eps=0.03, seed=0):
    # 训练集（50/50）
    X_train, y_train = make_train_on_line_balanced(n_total=train_total, eps=eps, seed=seed)

    # 测试集（尽量均匀的 50 个点）
    X_test_np = latin_hypercube_2d(test_n, seed=seed + 1)
    y_test_np = label_on_line(X_test_np, eps=eps)

    # 转 torch 并放设备
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = torch.from_numpy(X_test_np).to(device)
    y_test = torch.from_numpy(y_test_np).to(device)
    return X_train, y_train, X_test, y_test, eps

# ========== [2] 模型 ==========
def build_model():
    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)  # 输出 logit
    ).to(device)
    return model

# ========== [3] 训练与评估 ==========
def accuracy_from_logits(logits, y):
    pred = (torch.sigmoid(logits) > 0.5).float()
    return (pred.eq(y)).float().mean().item()

def train(model, X_train, y_train, epochs=1000, batch_size=256, lr=3e-3, weight_decay=1e-4, log_every=100):
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_hist, acc_hist = [], []
    N = X_train.size(0)

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(N, device=device)
        Xb_all, yb_all = X_train[idx], y_train[idx]

        for s in range(0, N, batch_size):
            xb = Xb_all[s:s + batch_size]
            yb = yb_all[s:s + batch_size]

            logits = model(xb)
            loss = criterion(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_hist.append(loss.item())

        # 记录训练准确率
        model.eval()
        with torch.no_grad():
            train_logits = model(X_train)
            train_acc = accuracy_from_logits(train_logits, y_train)
            acc_hist.append(train_acc)

        if epoch % log_every == 0:
            print(f"Epoch {epoch:04d} | train_acc {train_acc:.3f} | last_loss {loss.item():.4f}")

    return loss_hist, acc_hist

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        test_acc = accuracy_from_logits(logits, y_test)
    print(f"\n[Test] accuracy: {test_acc:.3f}")
    return test_acc

# ========== [4] 可视化（只显示测试集；背景 0→红、1→绿 渐变） ==========
def plot_decision_and_test_only(model, X_test, y_test, title_suffix=""):
    GRID_N = 400
    # 网格
    xx, yy = np.meshgrid(
        np.linspace(VIS_X_MIN, VIS_X_MAX, GRID_N),
        np.linspace(VIS_Y_MIN, VIS_Y_MAX
    GRID_N),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        zz = torch.sigmoid(model(grid)).reshape(xx.shape)  # 概率场
        probs = torch.sigmoid(model(X_test))
        preds = (probs > 0.5).float()

    # 背景：0→红，1→绿
    red_green = LinearSegmentedColormap.from_list("red_green", ["red", "green"])
    zz_np = to_cpu(zz).numpy()

    X_test_np = to_cpu(X_test).numpy()
    y_test_np = to_cpu(y_test).numpy().astype(int).reshape(-1)
    preds_np  = to_cpu(preds).numpy().astype(int).reshape(-1)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, zz_np, levels=100, cmap=red_green, alpha=0.6)
    plt.contour(xx, yy, zz_np, levels=[0.5], linewidths=2, colors="black")  # p=0.5 决策边界

    # 预测与真值是否一致：一致=紫色圆，不一致=黑色圆（空心）
    correct = preds_np == y_test_np
    plt.scatter(
        X_test_np[correct, 0], X_test_np[correct, 1],
        s=50, facecolors="none", edgecolors="purple", linewidths=1.6, label="Correct (purple ○)"
    )
    plt.scatter(
        X_test_np[~correct, 0], X_test_np[~correct, 1],
        s=50, facecolors="none", edgecolors="black", linewidths=1.6, label="Wrong (black ○)"
    )

    plt.title(f"Decision Field (red=0, green=1) {title_suffix}".strip())
    plt.xlabel("x"); plt.ylabel("y")
    plt.xlim(VIS_X_MIN, VIS_X_MAX); plt.ylim(VIS_Y_MIN, VIS_Y_MAX)
    plt.legend(loc="upper right", framealpha=0.85)
    plt.tight_layout(); plt.show()

def plot_curves(loss_hist, acc_hist):
    plt.figure(figsize=(6, 3))
    plt.plot(loss_hist, label="loss")
    plt.legend(); plt.title("Training loss"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(6, 3))
    plt.plot(acc_hist, label="train acc per epoch")
    plt.legend(); plt.title("Train accuracy"); plt.tight_layout(); plt.show()

# ========== [5] 主流程 ==========
if __name__ == "__main__":
    # 固定随机种子（可复现）
    torch.manual_seed(0)
    np.random.seed(0)

    # 数据：训练集 4000（50/50），测试集 50（均匀分布）；eps 为“在直线上的带宽”
    eps = 0.03
    X_train, y_train, X_test, y_test, _ = prepare_data(
        train_total=4000, test_n=50, eps=eps, seed=0
    )

    # 模型 & 训练
    model = build_model()
    loss_hist, acc_hist = train(
        model, X_train, y_train,
        epochs=1000, batch_size=256, lr=3e-3, weight_decay=1e-4, log_every=100
    )

    # 测试 & 可视化（仅测试集）
    _ = evaluate(model, X_test, y_test)
    plot_decision_and_test_only(
        model, X_test, y_test,
        title_suffix=f"— test only (eps={eps})"
    )
    plot_curves(loss_hist, acc_hist)
