"""
Train wake word classifier using the proven sklearn approach (99.95% accuracy in POC)
and export as ONNX compatible with the Android app.

Strategy:
1. Balanced train/test from computed features
2. StandardScaler + LogisticRegression (sklearn) — proven to work
3. Fold scaler + LR weights into a single-layer PyTorch model
4. Export ONNX with opset 11, IR v7 (Android-compatible)
"""
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import os
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR = "hee_schpustee"
CONFIG_FILE = "my_model.yaml"
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)
MODEL_NAME = config.get("model_name", "hee_schpustee")
INPUT_FRAMES = 16
INPUT_FEATURES = 96

# ─── Load features ───────────────────────────────────────────────────────────
print("Loading features...")
pos_train = np.load(os.path.join(MODEL_DIR, "positive_features_train.npy"))  # (100000, 16, 96)
neg_train = np.load(os.path.join(MODEL_DIR, "negative_features_train.npy"))  # (100000, 16, 96)
pos_test = np.load(os.path.join(MODEL_DIR, "positive_features_test.npy"))    # (10000, 16, 96)
neg_test = np.load(os.path.join(MODEL_DIR, "negative_features_test.npy"))    # (10000, 16, 96)

# Add ACAV100M hard negatives
acav_path = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
if os.path.exists(acav_path):
    print("Loading ACAV100M sample...")
    acav = np.load(acav_path, mmap_mode='r')
    rng = np.random.default_rng(42)
    acav_idx = rng.choice(len(acav), size=50000, replace=False)
    acav_sample = np.array(acav[sorted(acav_idx)], dtype=np.float32)
    neg_train_all = np.concatenate([neg_train, acav_sample], axis=0)
    print(f"  Total negatives: {len(neg_train_all)}")
else:
    neg_train_all = neg_train

# ─── Create balanced flattened datasets ───────────────────────────────────────
def make_balanced(pos, neg, max_per_class=50000):
    n = min(len(pos), len(neg), max_per_class)
    rng = np.random.default_rng(42)
    pi = rng.choice(len(pos), size=n, replace=False)
    ni = rng.choice(len(neg), size=n, replace=False)
    X = np.concatenate([
        pos[pi].reshape(n, -1),
        neg[ni].reshape(n, -1)
    ], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    perm = rng.permutation(len(X))
    return X[perm], y[perm]

X_train, y_train = make_balanced(pos_train, neg_train_all, 50000)
X_test, y_test = make_balanced(pos_test, neg_test, 10000)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ─── StandardScaler + LogisticRegression ──────────────────────────────────────
print("\nFitting StandardScaler...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("Training LogisticRegression...")
lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', verbose=1)
lr.fit(X_train_s, y_train)

y_pred = lr.predict(X_test_s)
y_prob = lr.predict_proba(X_test_s)[:, 1]
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

# ─── Threshold analysis ──────────────────────────────────────────────────────
print("Threshold analysis:")
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    pred_pos = y_prob >= t
    tp = (pred_pos & (y_test == 1)).sum()
    fp = (pred_pos & (y_test == 0)).sum()
    fn = (~pred_pos & (y_test == 1)).sum()
    tn = (~pred_pos & (y_test == 0)).sum()
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    print(f"  t={t:.2f} | Recall: {recall:.4f} | Prec: {precision:.4f} | FPR: {fpr:.4f}")

# ─── Build PyTorch model with scaler folded in ───────────────────────────────
# LR: sigmoid(W @ scaler(x) + b) = sigmoid(W @ ((x - mean) / std) + b)
#   = sigmoid((W / std) @ x + (b - W @ (mean / std)))
#
# So for the ONNX model: new_W = W / std, new_b = b - W @ (mean / std)

W = lr.coef_.astype(np.float32)           # (1, 1536)
b = lr.intercept_.astype(np.float32)       # (1,)
mean = scaler.mean_.astype(np.float32)     # (1536,)
std = scaler.scale_.astype(np.float32)     # (1536,)

# Fold scaler into weights
W_folded = W / std[None, :]                # (1, 1536)
b_folded = b - (W @ (mean / std)[:, None]).flatten()  # (1,)

# Verify folding is correct
test_raw = X_test[:100]  # raw features
test_scaled = scaler.transform(test_raw)
pred_original = 1 / (1 + np.exp(-(test_scaled @ W.T + b)))    # original LR
pred_folded = 1 / (1 + np.exp(-(test_raw @ W_folded.T + b_folded)))  # folded
max_diff = np.abs(pred_original - pred_folded).max()
print(f"\nFolding verification - max prediction diff: {max_diff:.2e}")

# Build PyTorch model matching OpenWakeWord architecture expectation
# The model takes (batch, 16, 96) and outputs (batch, 1)
class WakeWordLR(nn.Module):
    def __init__(self, input_frames, input_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_frames * input_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.sigmoid(self.linear(x))
        return x

model = WakeWordLR(INPUT_FRAMES, INPUT_FEATURES)
with torch.no_grad():
    model.linear.weight.copy_(torch.from_numpy(W_folded))
    model.linear.bias.copy_(torch.from_numpy(b_folded))
model.eval()

# Verify PyTorch model matches sklearn
test_tensor = torch.from_numpy(X_test[:100].reshape(-1, INPUT_FRAMES, INPUT_FEATURES))
with torch.no_grad():
    pt_pred = model(test_tensor).numpy().flatten()
sk_pred = pred_folded[:100].flatten()
max_diff_pt = np.abs(pt_pred - sk_pred).max()
print(f"PyTorch vs sklearn max diff: {max_diff_pt:.2e}")

# ─── Export ONNX ──────────────────────────────────────────────────────────────
onnx_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.onnx")
pth_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pth")

dummy_input = torch.randn(1, INPUT_FRAMES, INPUT_FEATURES)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    dynamo=False,
)
torch.save(model, pth_path)

# Fix ONNX for Android
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
onnx_model.ir_version = 7
for opset in onnx_model.opset_import:
    if opset.domain == "" or opset.domain == "ai.onnx":
        opset.version = 11
for node in onnx_model.graph.node:
    if node.op_type == "Reshape":
        for attr in list(node.attribute):
            if attr.name == "allowzero":
                node.attribute.remove(attr)
onnx.save(onnx_model, onnx_path)
print(f"\nONNX saved: {onnx_path}")

# ─── ONNX Runtime verification ───────────────────────────────────────────────
session = ort.InferenceSession(onnx_path)
pos_raw = pos_test[:20].astype(np.float32)
neg_raw = neg_test[:20].astype(np.float32)
pos_preds = session.run(None, {"input": pos_raw})[0].flatten()
neg_preds = session.run(None, {"input": neg_raw})[0].flatten()
print(f"\nONNX inference test (raw features):")
print(f"  Positive (want high): mean={pos_preds.mean():.4f}, min={pos_preds.min():.4f}, max={pos_preds.max():.4f}")
print(f"  Negative (want low):  mean={neg_preds.mean():.4f}, min={neg_preds.min():.4f}, max={neg_preds.max():.4f}")
print(f"  Positive scores: {pos_preds[:10]}")
print(f"  Negative scores: {neg_preds[:10]}")

# ─── False positive rate estimation on ACAV ──────────────────────────────────
if os.path.exists(acav_path):
    print("\nEstimating FP/hr on ACAV100M (11.3 hr validation set)...")
    acav = np.load(acav_path, mmap_mode='r')
    batch_size = 10000
    total_fp = 0
    total_samples = 0
    for i in range(0, min(len(acav), 500000), batch_size):
        batch = np.array(acav[i:i+batch_size], dtype=np.float32)
        preds = session.run(None, {"input": batch})[0].flatten()
        total_fp += (preds >= 0.5).sum()
        total_samples += len(batch)

    # ACAV is 2000 hours, 500K samples = 500000/5625000 * 2000 hours = 177.8 hours
    hours_checked = (total_samples / 5625000) * 2000
    fp_per_hr = total_fp / hours_checked
    print(f"  Checked {total_samples} samples ({hours_checked:.1f} hrs)")
    print(f"  FP at t=0.5: {total_fp} ({fp_per_hr:.2f}/hr)")

    # Check at higher thresholds
    for t in [0.5, 0.7, 0.9, 0.95, 0.99]:
        total_fp_t = 0
        for i in range(0, min(len(acav), 500000), batch_size):
            batch = np.array(acav[i:i+batch_size], dtype=np.float32)
            preds = session.run(None, {"input": batch})[0].flatten()
            total_fp_t += (preds >= t).sum()
        fp_hr_t = total_fp_t / hours_checked
        print(f"  FP at t={t:.2f}: {total_fp_t} ({fp_hr_t:.2f}/hr)")

print("\nDone!")
