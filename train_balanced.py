"""
Direct balanced training of wake word classifier.
Bypasses OpenWakeWord's complex weight scheduling that causes model collapse.
Trains with balanced batches and exports ONNX compatible with the Android app.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import yaml
import onnx

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR = "hee_schpustee"
CONFIG_FILE = "my_model.yaml"

with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

LAYER_DIM = config.get("layer_size", 128)
INPUT_FRAMES = 16  # total_length=32000 / 2000 = 16 frames
INPUT_FEATURES = 96  # openWakeWord embedding dim
MODEL_NAME = config.get("model_name", "hee_schpustee")

print(f"Config: layer_dim={LAYER_DIM}, input=({INPUT_FRAMES}, {INPUT_FEATURES})")

# ─── Model (exact same architecture as OpenWakeWord train.py) ─────────────────
class FCNBlock(nn.Module):
    def __init__(self, layer_dim):
        super().__init__()
        self.fcn_layer = nn.Linear(layer_dim, layer_dim)
        self.batch_norm = nn.BatchNorm1d(layer_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batch_norm(self.fcn_layer(x)))


class Net(nn.Module):
    def __init__(self, input_shape, layer_dim, n_blocks=1, n_classes=1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_shape[0] * input_shape[1], layer_dim)
        self.batch_norm1 = nn.BatchNorm1d(layer_dim)
        self.relu1 = nn.ReLU()
        self.blocks = nn.ModuleList([FCNBlock(layer_dim) for _ in range(n_blocks)])
        self.last_layer = nn.Linear(layer_dim, n_classes)
        self.last_act = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.batch_norm1(self.layer1(self.flatten(x))))
        for block in self.blocks:
            x = block(x)
        x = self.last_act(self.last_layer(x))
        return x


# ─── Load features ───────────────────────────────────────────────────────────
print("Loading features...")
pos_train = np.load(os.path.join(MODEL_DIR, "positive_features_train.npy"))
neg_train = np.load(os.path.join(MODEL_DIR, "negative_features_train.npy"))
pos_test = np.load(os.path.join(MODEL_DIR, "positive_features_test.npy"))
neg_test = np.load(os.path.join(MODEL_DIR, "negative_features_test.npy"))

# Also load ACAV100M for harder negatives (random sample)
acav_path = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
if os.path.exists(acav_path):
    print("Loading ACAV100M sample for hard negatives...")
    acav = np.load(acav_path, mmap_mode='r')
    rng = np.random.default_rng(42)
    acav_indices = rng.choice(len(acav), size=min(50000, len(acav)), replace=False)
    acav_sample = np.array(acav[sorted(acav_indices)], dtype=np.float32)
    neg_train = np.concatenate([neg_train, acav_sample], axis=0)
    print(f"  Added {len(acav_sample)} ACAV negatives -> total neg train: {len(neg_train)}")

print(f"Positive train: {pos_train.shape}, Negative train: {neg_train.shape}")
print(f"Positive test: {pos_test.shape}, Negative test: {neg_test.shape}")

# ─── Create balanced dataset ─────────────────────────────────────────────────
def make_balanced_dataset(pos, neg, max_samples_per_class=50000):
    """Create a balanced dataset by undersampling the majority class."""
    n = min(len(pos), len(neg), max_samples_per_class)
    rng = np.random.default_rng(42)

    pos_idx = rng.choice(len(pos), size=n, replace=len(pos) < n)
    neg_idx = rng.choice(len(neg), size=n, replace=len(neg) < n)

    X = np.concatenate([pos[pos_idx], neg[neg_idx]], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(n), np.zeros(n)], axis=0).astype(np.float32)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


X_train, y_train = make_balanced_dataset(pos_train, neg_train, max_samples_per_class=50000)
X_test, y_test = make_balanced_dataset(pos_test, neg_test, max_samples_per_class=10000)
print(f"Balanced train: {X_train.shape}, pos={y_train.sum():.0f}, neg={(1-y_train).sum():.0f}")
print(f"Balanced test: {X_test.shape}, pos={y_test.sum():.0f}, neg={(1-y_test).sum():.0f}")

# ─── DataLoaders ──────────────────────────────────────────────────────────────
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)

# ─── Train ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = Net(
    input_shape=(INPUT_FRAMES, INPUT_FEATURES),
    layer_dim=LAYER_DIM,
    n_blocks=1,
    n_classes=1
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
loss_fn = nn.BCELoss()

EPOCHS = 30
best_acc = 0
best_state = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    n_batches = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch).squeeze()
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    scheduler.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch).squeeze()
            predicted = (pred >= 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            tp += ((predicted == 1) & (y_batch == 1)).sum().item()
            fp += ((predicted == 1) & (y_batch == 0)).sum().item()
            fn += ((predicted == 0) & (y_batch == 1)).sum().item()
            tn += ((predicted == 0) & (y_batch == 0)).sum().item()

    acc = correct / total
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/n_batches:.4f} | "
          f"Acc: {acc:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | "
          f"FPR: {fpr:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

# Load best model
model.load_state_dict(best_state)
print(f"\nBest accuracy: {best_acc:.4f}")

# ─── Final evaluation at multiple thresholds ─────────────────────────────────
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        pred = model(X_batch).squeeze().cpu().numpy()
        all_preds.append(pred)
        all_labels.append(y_batch.numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

print("\nThreshold analysis:")
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred_pos = all_preds >= t
    tp = ((pred_pos) & (all_labels == 1)).sum()
    fp = ((pred_pos) & (all_labels == 0)).sum()
    fn = ((~pred_pos) & (all_labels == 1)).sum()
    tn = ((~pred_pos) & (all_labels == 0)).sum()
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    print(f"  t={t:.1f} | Recall: {recall:.4f} | Precision: {precision:.4f} | FPR: {fpr:.4f}")

# ─── Export ONNX ──────────────────────────────────────────────────────────────
model.eval()
model.cpu()
dummy_input = torch.randn(1, INPUT_FRAMES, INPUT_FEATURES)
onnx_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.onnx")
pth_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pth")

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

# Verify ONNX
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# Fix opset and IR version for Android compatibility
onnx_model.ir_version = 7
for opset in onnx_model.opset_import:
    if opset.domain == "" or opset.domain == "ai.onnx":
        opset.version = 11

# Remove allowzero attributes from Reshape nodes
for node in onnx_model.graph.node:
    if node.op_type == "Reshape":
        for attr in list(node.attribute):
            if attr.name == "allowzero":
                node.attribute.remove(attr)

onnx.save(onnx_model, onnx_path)
print(f"\nONNX model saved to: {onnx_path}")
print(f"PTH model saved to: {pth_path}")

# Quick inference test
import onnxruntime as ort
session = ort.InferenceSession(onnx_path)
test_pos = pos_test[:10].astype(np.float32)
test_neg = neg_test[:10].astype(np.float32)
pos_preds = session.run(None, {"input": test_pos})[0]
neg_preds = session.run(None, {"input": test_neg})[0]
print(f"\nONNX inference test:")
print(f"  Positive samples (should be high): {pos_preds.flatten()}")
print(f"  Negative samples (should be low):  {neg_preds.flatten()}")
