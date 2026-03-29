"""
v3 training: iterative hard negative mining with focal loss and deeper network.
Strategy:
  1. Start with ACAV-only training (random negatives)
  2. Score ACAV, find hard negatives
  3. Retrain with hard negatives heavily oversampled + focal loss
  4. Repeat mining 2 more times
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

# --- Focal loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()

# --- Deeper model ---
class WakeWordDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1536, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train_model(X_train, y_train, X_test, y_test, epochs=40, use_focal=True):
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    train_dl = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=2048)

    model = WakeWordDNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0) if use_focal else nn.BCELoss()
    
    best_acc = 0
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        scheduler.step()
        
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                all_preds.append(model(xb.to(device)).squeeze().cpu())
                all_labels.append(yb)
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        acc = accuracy_score(labels, (preds > 0.5).astype(int))
        recall = ((preds > 0.5) & (labels == 1)).sum() / max((labels == 1).sum(), 1)
        fpr = ((preds > 0.5) & (labels == 0)).sum() / max((labels == 0).sum(), 1)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f'  Epoch {epoch+1:2d} | Loss={total_loss/len(train_dl):.4f} | Acc={acc:.4f} | Recall={recall:.4f} | FPR={fpr:.4f}', flush=True)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    return model, best_acc

# Load data
print('Loading features...', flush=True)
pos_train = np.load('hee_schpustee/positive_features_train.npy').reshape(-1, 1536)
pos_test = np.load('hee_schpustee/positive_features_test.npy').reshape(-1, 1536)
acav = np.load('openwakeword_features_ACAV100M_2000_hrs_16bit.npy', mmap_mode='r')
n_acav = len(acav)
rng = np.random.default_rng(42)

# Random negatives for baseline + test
random_idxs = rng.choice(n_acav, size=150000, replace=False)
print('Loading initial random ACAV negatives...', flush=True)
acav_random = np.array([acav[i].flatten().astype(np.float32) for i in sorted(random_idxs[:100000])])
acav_test = np.array([acav[i].flatten().astype(np.float32) for i in sorted(random_idxs[100000:110000])])

X_test = np.concatenate([pos_test[:10000], acav_test])
y_test = np.concatenate([np.ones(10000), np.zeros(10000)])

# ============ ROUND 1: Random ACAV negatives with focal loss ============
print('\n=== ROUND 1: Random ACAV + Focal Loss ===', flush=True)
n_use = min(len(pos_train), len(acav_random))
X_train_r1 = np.concatenate([pos_train[:n_use], acav_random[:n_use]])
y_train_r1 = np.concatenate([np.ones(n_use), np.zeros(n_use)])

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_r1)
X_test_s = scaler.transform(X_test)

perm = rng.permutation(len(X_train_s))
X_train_s, y_train_r1 = X_train_s[perm], y_train_r1[perm]

model_r1, acc_r1 = train_model(X_train_s, y_train_r1, X_test_s, y_test, epochs=30, use_focal=True)
print(f'Round 1 best accuracy: {acc_r1:.4f}', flush=True)

# ============ MINE HARD NEGATIVES (Round 1) ============
print('\n--- Mining hard negatives with round 1 model ---', flush=True)
model_r1.eval()
hard_neg_indices = []
batch_size = 50000
for start in range(0, min(2000000, n_acav), batch_size):
    end = min(start + batch_size, n_acav)
    chunk = np.ascontiguousarray(acav[start:end]).astype(np.float32).reshape(-1, 1536)
    chunk_s = scaler.transform(chunk)
    with torch.no_grad():
        scores = model_r1(torch.FloatTensor(chunk_s).to(device)).squeeze().cpu().numpy()
    for j in range(len(scores)):
        if scores[j] > 0.3:
            hard_neg_indices.append(start + j)
    if start % 500000 == 0:
        print(f'  Scanned {start}/2000000 ({len(hard_neg_indices)} hard negs)...', flush=True)

print(f'Found {len(hard_neg_indices)} hard negatives (score>0.3 from 2M ACAV)', flush=True)

if len(hard_neg_indices) > 0:
    hard_neg_feats = np.array([acav[i].flatten().astype(np.float32) for i in hard_neg_indices])
else:
    hard_neg_feats = np.empty((0, 1536))

# ============ ROUND 2: Hard negs oversampled + focal loss ============
print(f'\n=== ROUND 2: {len(hard_neg_feats)} hard negs + random ACAV + Focal Loss ===', flush=True)
n_hard = len(hard_neg_feats)
if n_hard > 0:
    # Heavy oversample: make hard negatives 50% of all negatives
    n_hard_target = min(100000, n_hard * max(1, 100000 // max(n_hard, 1)))
    hard_neg_oversampled = hard_neg_feats[rng.choice(n_hard, size=n_hard_target, replace=True)] if n_hard < n_hard_target else hard_neg_feats[:n_hard_target]
    neg_train_r2 = np.concatenate([acav_random[:100000], hard_neg_oversampled])
else:
    neg_train_r2 = acav_random[:100000]

n_neg_r2 = len(neg_train_r2)
n_pos_r2 = min(len(pos_train), n_neg_r2)
X_train_r2 = np.concatenate([np.tile(pos_train[:100000], (max(1, n_neg_r2 // 100000), 1))[:n_neg_r2], neg_train_r2])
y_train_r2 = np.concatenate([np.ones(n_neg_r2), np.zeros(n_neg_r2)])

# Refit scaler on round 2 data
scaler2 = StandardScaler()
X_train_s2 = scaler2.fit_transform(X_train_r2)
X_test_s2 = scaler2.transform(X_test)

perm = rng.permutation(len(X_train_s2))
X_train_s2, y_train_r2 = X_train_s2[perm], y_train_r2[perm]

model_r2, acc_r2 = train_model(X_train_s2, y_train_r2, X_test_s2, y_test, epochs=40, use_focal=True)
print(f'Round 2 best accuracy: {acc_r2:.4f}', flush=True)

# ============ MINE HARD NEGATIVES (Round 2) ============
print('\n--- Mining hard negatives with round 2 model ---', flush=True)
model_r2.eval()
hard_neg_indices_r2 = []
for start in range(0, min(2000000, n_acav), batch_size):
    end = min(start + batch_size, n_acav)
    chunk = np.ascontiguousarray(acav[start:end]).astype(np.float32).reshape(-1, 1536)
    chunk_s = scaler2.transform(chunk)
    with torch.no_grad():
        scores = model_r2(torch.FloatTensor(chunk_s).to(device)).squeeze().cpu().numpy()
    for j in range(len(scores)):
        if scores[j] > 0.3:
            hard_neg_indices_r2.append(start + j)
    if start % 500000 == 0:
        print(f'  Scanned {start}/2000000 ({len(hard_neg_indices_r2)} hard negs)...', flush=True)

print(f'Found {len(hard_neg_indices_r2)} hard negatives round 2', flush=True)

if len(hard_neg_indices_r2) > 0:
    hard_neg_feats_r2 = np.array([acav[i].flatten().astype(np.float32) for i in hard_neg_indices_r2])
else:
    hard_neg_feats_r2 = np.empty((0, 1536))

# ============ ROUND 3: All hard negs + focal loss ============
all_hard = []
if n_hard > 0: all_hard.append(hard_neg_feats)
if len(hard_neg_feats_r2) > 0: all_hard.append(hard_neg_feats_r2)
if all_hard:
    all_hard_negs = np.unique(np.concatenate(all_hard), axis=0)
else:
    all_hard_negs = np.empty((0, 1536))

print(f'\n=== ROUND 3: {len(all_hard_negs)} unique hard negs + random ACAV + Focal Loss ===', flush=True)
n_all_hard = len(all_hard_negs)
if n_all_hard > 0:
    n_hard_target3 = min(150000, n_all_hard * max(1, 150000 // max(n_all_hard, 1)))
    hard_neg_os3 = all_hard_negs[rng.choice(n_all_hard, size=n_hard_target3, replace=True)] if n_all_hard < n_hard_target3 else all_hard_negs[:n_hard_target3]
    neg_train_r3 = np.concatenate([acav_random[:100000], hard_neg_os3])
else:
    neg_train_r3 = acav_random[:100000]

n_neg_r3 = len(neg_train_r3)
X_train_r3 = np.concatenate([np.tile(pos_train[:100000], (max(1, n_neg_r3 // 100000 + 1), 1))[:n_neg_r3], neg_train_r3])
y_train_r3 = np.concatenate([np.ones(n_neg_r3), np.zeros(n_neg_r3)])

scaler3 = StandardScaler()
X_train_s3 = scaler3.fit_transform(X_train_r3)
X_test_s3 = scaler3.transform(X_test)

perm = rng.permutation(len(X_train_s3))
X_train_s3, y_train_r3 = X_train_s3[perm], y_train_r3[perm]

model_r3, acc_r3 = train_model(X_train_s3, y_train_r3, X_test_s3, y_test, epochs=50, use_focal=True)
print(f'Round 3 best accuracy: {acc_r3:.4f}', flush=True)

# Save best model + scaler
torch.save({'model': model_r3.state_dict(), 'mean': scaler3.mean_, 'std': scaler3.scale_}, 'hee_schpustee/best_dnn_v3.pth')
np.savez('hee_schpustee/scaler_v3.npz', mean=scaler3.mean_, std=scaler3.scale_)
print('\nSaved best_dnn_v3.pth + scaler_v3.npz', flush=True)

# Quick FP check on the 2M ACAV subset we already scanned
print('\n--- Quick FP check on 500K ACAV ---', flush=True)
model_r3.eval()
fp_counts = {t: 0 for t in [0.5, 0.9, 0.99, 0.999, 0.9999]}
n_check = 500000
for start in range(0, n_check, batch_size):
    end = min(start + batch_size, n_check)
    chunk = np.ascontiguousarray(acav[start:end]).astype(np.float32).reshape(-1, 1536)
    chunk_s = scaler3.transform(chunk)
    with torch.no_grad():
        scores = model_r3(torch.FloatTensor(chunk_s).to(device)).squeeze().cpu().numpy()
    for t in fp_counts:
        fp_counts[t] += int((scores >= t).sum())

hrs = n_check * 1.28 / 3600
print(f'FP/hr over {hrs:.0f} hrs:')
for t, n in sorted(fp_counts.items()):
    print(f'  t={t}: {n} FPs = {n/hrs:.1f}/hr')

# Positive recall
pos_s = scaler3.transform(pos_test[:5000].reshape(-1, 1536))
with torch.no_grad():
    ps = model_r3(torch.FloatTensor(pos_s).to(device)).squeeze().cpu().numpy()
for t in [0.5, 0.9, 0.99, 0.999, 0.9999]:
    print(f'  Pos recall t={t}: {(ps>=t).mean():.4f}')
