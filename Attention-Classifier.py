# Multi-pair-based with bottom 50% filtering removal

from itertools import combinations
import numpy as np
import os
import pandas as pd
from scipy.stats import norm
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

url = "https://raw.githubusercontent.com/aaravdaftary/Attention-Bias-Neuron-Comparison/refs/heads/main/Engagement%20Data%20-%20Sheet17.csv"

df = pd.read_csv(url)
df.columns = df.columns.str.strip().str.lower()

df["clicks"] = pd.to_numeric(df["clicks"])
df["impressions"] = pd.to_numeric(df["impressions"])
df["ctr"] = df["clicks"] / df["impressions"]

sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

embeddings = sbert.encode(
    df["headline"].astype(str).tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)

df["embedding"] = list(embeddings)

pair_data = []

for exp_id, group in df.groupby("clickability_test_id"):
    group = group.reset_index(drop=True)
    for i, j in combinations(range(len(group)), 2):
        row_i = group.iloc[i]
        row_j = group.iloc[j]

        pair_data.append(
            {
                "experiment": exp_id,
                "ctr_a": row_i["ctr"],
                "ctr_b": row_j["ctr"],
                "emb_a": row_i["embedding"],
                "emb_b": row_j["embedding"],
                "clicks_a": row_i["clicks"],
                "clicks_b": row_j["clicks"],
                "impr_a": row_i["impressions"],
                "impr_b": row_j["impressions"],
            }
        )

pair_df = pd.DataFrame(pair_data)


def pair_z_score(c1, n1, c2, n2):
    p1 = c1 / n1
    p2 = c2 / n2
    pooled = (c1 + c2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0
    return abs((p1 - p2) / se)

expanded = []

for _, row in pair_df.iterrows():
    z = pair_z_score(
        row["clicks_a"], row["impr_a"], row["clicks_b"], row["impr_b"]
    )

    if row["ctr_a"] > row["ctr_b"]:
        expanded.append(
            {
                "emb_diff": row["emb_a"] - row["emb_b"],
                "label": 1,
                "weight": z,
            }
        )
        expanded.append(
            {
                "emb_diff": row["emb_b"] - row["emb_a"],
                "label": 0,
                "weight": z,
            }
        )

    else:
        expanded.append(
            {
                "emb_diff": row["emb_b"] - row["emb_a"],
                "label": 1,
                "weight": z,
            }
        )
        expanded.append(
            {
                "emb_diff": row["emb_a"] - row["emb_b"],
                "label": 0,
                "weight": z,
            }
        )


expanded_df = pd.DataFrame(expanded)

class PairDataset(Dataset):

    def __init__(self, X, y, weights):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]


class LogisticClassifier(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)

class MLPClassifier(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(0.20),


            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.20),


            nn.Linear(128, 1)
        )


    def forward(self, x):

        return self.network(x).squeeze(1)

def evaluate(model, X_tensor, y_true):
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).cpu().numpy()
    return f1_score(y_true, preds)


def train_one_fold(
    X_train,
    y_train,
    w_train,
    X_val,
    y_val,
    epochs=60,
    batch_size=32,
    lr=2e-4,
    weight_decay=0,
    save_path="best_model.pth",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch_f1s = []

    model = MLPClassifier(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer,
    #    mode="max",
    #    factor=0.25,
    #    patience=20,
    #    min_lr=lr*0.005
    #)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    train_loader = DataLoader(
        PairDataset(X_train, y_train, w_train),
        batch_size=batch_size,
        shuffle=True,
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_f1 = 0
    epochs_without_improve = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            logits = model(xb)

            loss = criterion(logits, yb)
            loss = (loss * wb).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_f1 = evaluate(model, X_train_tensor, y_train)
        val_f1 = evaluate(model, X_val_tensor, y_val)
        # scheduler.step(val_f1)



        # best_f1 = max(best_f1, val_f1)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_f1s.append(val_f1)
        if val_f1 > best_f1:
            print(
                f"Epoch {epoch+1:02d} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Train F1: {train_f1:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"LR: {current_lr:.6f} | improved over best F1: {best_f1:.4f}"
            )
            best_f1 = val_f1
            epochs_without_improve = 0

            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path}")

        else:
            epochs_without_improve += 1
            #print(
            #    f"Epoch {epoch+1:02d} | "
            #    f"Loss: {epoch_loss:.4f} | "
            #    f"Train F1: {train_f1:.4f} | "
            #    f"Val F1: {val_f1:.4f} | no improve: {epochs_without_improve:02d}"
            #)

        if epochs_without_improve >= 10:
            break

    epoch_f1s = np.array(epoch_f1s)

    if len(epoch_f1s) < epochs:
        epoch_f1s = np.pad(
            epoch_f1s,
            (0, epochs - len(epoch_f1s)),
            constant_values=np.nan
        )

    return epoch_f1s

# percentiles = [0.8, 0.805, 0.81, 0.815, 0.82, 0.825, 0.83, 0.835, 0.84, 0.845, 0.85]
# 0.825=best perf,0.835=best stablity
# wd marginal benefit. best at 0,1e-5

for wd in [1e-5]:
    print("=" * 50)
    print(f"Testing wd: {wd}")

    z_threshold = expanded_df["weight"].quantile(0.825)

    expanded_df_p = expanded_df[expanded_df["weight"] >= z_threshold].reset_index(drop=True)

    print(f"Original expanded pairs count: {len(expanded)}")
    print(f"Filtered count (Top %ile Z-score >= {z_threshold:.4f}): {len(expanded_df_p)}")
    # print(expanded_df_p.groupby("label")["weight"].describe())
    # print(expanded_df_p["label"].value_counts(normalize=True))
    # continue

    X = np.vstack(expanded_df_p["emb_diff"].values)
    y = expanded_df_p["label"].values.astype(np.int64)
    weights = expanded_df_p["weight"].values.astype(np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        print("\n")
        print("=" * 40)
        print(f"FOLD {fold+1}")
        print("=" * 40)

        X_train = X[train_idx]
        y_train = y[train_idx]
        w_train = weights[train_idx]

        X_val = X[val_idx]
        y_val = y[val_idx]

        fold_f1s = train_one_fold(
            X_train,
            y_train,
            w_train,
            X_val,
            y_val,
            epochs=80,
            batch_size=32,
            lr=2e-4,
            weight_decay=wd,
        )

        all_fold_f1s.append(fold_f1s)

    f1_matrix = np.array(all_fold_f1s)

    #print(f1_matrix.shape)

    #mean_f1 = np.nanmean(f1_matrix, axis=0)
    #std_f1 = np.nanstd(f1_matrix, axis=0)
    best_per_fold = np.nanmax(f1_matrix, axis=1)

    print("\nFinal Results")
    print("----------------")
    #print("Mean F1:", mean_f1)
    #print("Std F1:", std_f1)
    print("Mean best F1:", best_per_fold.mean())
    print("Std best F1 :", best_per_fold.std())

# TODO
# ReduceLROnPlateau
# weight_decay=1e-4,5e-4,1e-5
# dropout=0.2,0.25,0.3
