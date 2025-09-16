# xor_compare_data_quality_with_panel.py
# 三种训练集对比（A/B/C）+ 单图三联可视化：
# 背景：0→红, 1→绿 概率渐变；p=0.5 黑色等值线；测试点 50（LHS 均匀）
# 预测=真值 → 紫色空心圆；预测≠真值 → 黑色空心圆

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ========== 全局设置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
to_cpu = lambda t: t.detach().cpu()

# 可视化/采样范围
VIS_X_MIN, VIS_X_MAX = -0.5, 1.5
VIS_Y_MIN, VIS_Y_MAX = -0.5, 1.5

# 带宽：定义“在直线 y=x 上”
EPS_BAND = 0.03

# 训练超参
EPOCHS = 1000
BATCH_SIZE = 256
LR = 3e-3
WD = 1e-4
LOG_EVERY = 200

# ========== 公共工具 ==========
def latin_hypercube_2d(n, seed=0):
    rng = np.random.default_rng(seed)
    xs = (np.arange(n) + rng.random(n)) / n
    ys = (np.arange(n) + rng.random(n)) / n
    rng.shuffle(xs); rng.shuffle(ys)
    X = VIS_X_MIN + xs * (VIS_X_MAX - VIS_X_MIN)
    Y = VIS_Y_MIN + ys * (VIS_Y_MAX - VIS_Y_MIN)
    return np.stack([X, Y], axis=1).astype(np.float32)

def label_on_line(X_np, eps=EPS_BAND):
    return (np.abs(X_np[:, 0] - X_np[:, 1]) < eps).astype(np.float32).reshape(-1, 1)

def build_model():
    m = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)  # logit
    ).to(device)
    return m

def accuracy_from_logits(logits, y):
    pred = (torch.sigmoid(logits) > 0.5).float()
    return (pred.eq(y)).float().mean().item()

def train_one(model, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, wd=WD, log_every=LOG_EVERY):
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_hist, acc_hist = [], []
    N = X_train.size(0)

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(N, device=device)
        Xb_all, yb_all = X_train[idx], y_train[idx]

        for s in range(0, N, batch_size):
            xb = Xb_all[s:s+batch_size]
            yb = yb_all[s:s+batch_size]

            logits = model(xb)
            loss = criterion(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_hist.append(loss.item())

        model.eval()
        with torch.no_grad():
            train_logits = model(X_train)
            train_acc = accuracy_from_logits(train_logits, y_train)
            acc_hist.append(train_acc)

        if epoch % log_every == 0:
            print(f"  Epoch {epoch:04d} | train_acc {train_acc:.3f} | last_loss {loss.item():.4f}")

    return loss_hist, acc_hist

# ========== 三种训练集 ==========
def dataset_A_four_points(seed=0, eps=EPS_BAND):
    """A: 仅四个点。标签由 |x-y|<eps 判定。"""
    X_np = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    y_np = label_on_line(X_np, eps=eps)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_np))
    X_np, y_np = X_np[idx], y_np[idx]
    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    return X, y

def dataset_B_corners_perturbed(n_per_corner=200, noise=0.06, seed=0, eps=EPS_BAND):
    """B: 四点附近加噪。"""
    rng = np.random.default_rng(seed)
    centers = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    Xs = []
    for cx, cy in centers:
        local = rng.normal(loc=0.0, scale=noise, size=(n_per_corner, 2)).astype(np.float32)
        pts = local + np.array([cx, cy], dtype=np.float32)
        Xs.append(pts)
    X_np = np.vstack(Xs).astype(np.float32)
    y_np = label_on_line(X_np, eps=eps)
    idx = rng.permutation(X_np.shape[0])
    X_np, y_np = X_np[idx], y_np[idx]
    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    return X, y

def dataset_C_line_band_balanced(n_total=4000, eps=EPS_BAND, seed=0):
    """C: 50/50 均衡：一半 |x-y|<eps，一半在外。"""
    assert n_total % 2 == 0
    rng = np.random.default_rng(seed)
    n_pos = n_total // 2
    n_neg = n_total // 2

    # 正类（带宽内）
    u = rng.uniform(VIS_X_MIN, VIS_X_MAX, size=(n_pos, 1)).astype(np.float32)
    d = rng.uniform(-eps*0.98, eps*0.98, size=(n_pos, 1)).astype(np.float32)
    X_pos = np.hstack([u, u + d]).astype(np.float32)

    # 负类（带宽外，拒绝采样）
    X_neg = []
    need = n_neg
    while need > 0:
        cand = rng.uniform([VIS_X_MIN, VIS_Y_MIN], [VIS_X_MAX, VIS_Y_MAX], size=(need*2, 2)).astype(np.float32)
        mask = np.abs(cand[:,0] - cand[:,1]) >= eps
        picked = cand[mask]
        if picked.shape[0] > need: picked = picked[:need]
        X_neg.append(picked); need -= picked.shape[0]
    X_neg = np.concatenate(X_neg, axis=0).astype(np.float32)

    X_np = np.vstack([X_pos, X_neg]).astype(np.float32)
    y_np = np.vstack([np.ones((n_pos,1), np.float32), np.zeros((n_neg,1), np.float32)])
    idx = rng.permutation(X_np.shape[0])
    X_np, y_np = X_np[idx], y_np[idx]
    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    return X, y

# ========== 可视化：单图三联 ==========
def draw_field_on_ax(ax, model, X_test, y_test, title):
    """在给定 ax 上绘制背景概率场 + p=0.5 边界 + 测试点对错。"""
    model.eval()
    GRID_N = 400
    xx, yy = np.meshgrid(
        np.linspace(VIS_X_MIN, VIS_X_MAX, GRID_N),
        np.linspace(VIS_Y_MIN, VIS_Y_MAX, GRID_N)
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)

    with torch.no_grad():
        zz = torch.sigmoid(model(grid)).reshape(xx.shape)
        probs = torch.sigmoid(model(X_test))
        preds = (probs > 0.5).float()

    zz_np = to_cpu(zz).numpy()
    X_test_np = to_cpu(X_test).numpy()
    y_test_np = to_cpu(y_test).numpy().astype(int).reshape(-1)
    preds_np  = to_cpu(preds).numpy().astype(int).reshape(-1)

    red_green = LinearSegmentedColormap.from_list("red_green", ["red", "green"])
    cf = ax.contourf(xx, yy, zz_np, levels=100, cmap=red_green, alpha=0.6)
    ax.contour(xx, yy, zz_np, levels=[0.5], linewidths=2, colors="black")

    correct = preds_np == y_test_np
    ax.scatter(
        X_test_np[correct, 0], X_test_np[correct, 1],
        s=40, facecolors="none", edgecolors="purple", linewidths=1.6, label="Correct (purple ○)"
    )
    ax.scatter(
        X_test_np[~correct, 0], X_test_np[~correct, 1],
        s=40, facecolors="none", edgecolors="black", linewidths=1.6, label="Wrong (black ○)"
    )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(VIS_X_MIN, VIS_X_MAX); ax.set_ylim(VIS_Y_MIN, VIS_Y_MAX)
    return cf  # 返回用于统一 colorbar

def plot_compare_three(models, titles, X_test, y_test, accs):
    """1×3 子图对比三种训练集的决策区域。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    cfs = []
    for ax, model, title, acc in zip(axes, models, titles, accs):
        cf = draw_field_on_ax(ax, model, X_test, y_test, f"{title}\nTest Acc: {acc:.3f}")
        cfs.append(cf)

    # 统一放一个 colorbar（表示 P(class=1)）
    cbar = fig.colorbar(cfs[0], ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    cbar.set_label("P(label=1 | x)", rotation=90)

    # 只放一个图例（右下角）
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    plt.show()

# ========== 主流程 ==========
if __name__ == "__main__":
    # 固定随机种子
    torch.manual_seed(0); np.random.seed(0)

    # 同一测试集（50 个均匀分布点）
    X_test_np = latin_hypercube_2d(50, seed=2025)
    y_test_np = label_on_line(X_test_np, eps=EPS_BAND)
    X_test = torch.from_numpy(X_test_np).to(device)
    y_test = torch.from_numpy(y_test_np).to(device)

    models, titles, accs = [], [], []

    # ---- 方案 A ----
    print("\n=== Dataset A: Only four points ===")
    X_train_A, y_train_A = dataset_A_four_points(seed=1, eps=EPS_BAND)
    model_A = build_model()
    _ = train_one(model_A, X_train_A, y_train_A)
    with torch.no_grad():
        acc_A = accuracy_from_logits(model_A(X_test), y_test)
    print(f"Dataset A test acc = {acc_A:.3f}")
    models.append(model_A); titles.append("A) Four points"); accs.append(acc_A)

    # ---- 方案 B ----
    print("\n=== Dataset B: Perturbed around four corners ===")
    X_train_B, y_train_B = dataset_B_corners_perturbed(n_per_corner=200, noise=0.06, seed=2, eps=EPS_BAND)
    model_B = build_model()
    _ = train_one(model_B, X_train_B, y_train_B)
    with torch.no_grad():
        acc_B = accuracy_from_logits(model_B(X_test), y_test)
    print(f"Dataset B test acc = {acc_B:.3f}")
    models.append(model_B); titles.append("B) Perturbed corners"); accs.append(acc_B)

    # ---- 方案 C ----
    print("\n=== Dataset C: Balanced line-band (50/50) ===")
    X_train_C, y_train_C = dataset_C_line_band_balanced(n_total=4000, eps=EPS_BAND, seed=3)
    model_C = build_model()
    _ = train_one(model_C, X_train_C, y_train_C)
    with torch.no_grad():
        acc_C = accuracy_from_logits(model_C(X_test), y_test)
    print(f"Dataset C test acc = {acc_C:.3f}")
    models.append(model_C); titles.append("C) Balanced line-band"); accs.append(acc_C)

    # ---- 单图三联对比可视化 ----
    plot_compare_three(models, titles, X_test, y_test, accs)

    # ---- 控制台汇总 ----
    print("\n=== Summary (same test set, eps=%.3f) ===" % EPS_BAND)
    for name, Xtr, acc in zip(titles, [X_train_A, X_train_B, X_train_C], accs):
        print(f"{name:>26} | train N = {Xtr.size(0):5d} | test acc = {acc:.3f}")
