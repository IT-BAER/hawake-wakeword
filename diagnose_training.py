"""Diagnose why the wake word model isn't learning."""
import numpy as np
import torch
import os
import sys

# Add openwakeword to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "openwakeword"))

WORK_DIR = r"D:\VSC\hawake-wakeword\hee_schpustee"

print("=" * 60)
print("1. Feature file shapes")
print("=" * 60)
for fname in ["positive_features_test.npy", "negative_features_test.npy",
              "positive_features_train.npy", "negative_features_train.npy"]:
    path = os.path.join(WORK_DIR, fname)
    if os.path.exists(path):
        arr = np.load(path, mmap_mode='r')
        print(f"  {fname}: shape={arr.shape}, dtype={arr.dtype}")
        print(f"    min={arr[:100].min():.4f}, max={arr[:100].max():.4f}, mean={arr[:100].mean():.4f}")
    else:
        print(f"  {fname}: NOT FOUND")

print("\n" + "=" * 60)
print("2. Model output on positive test examples")
print("=" * 60)
pth_path = os.path.join(WORK_DIR, "hee_schpustee.pth")
if not os.path.exists(pth_path):
    print("  No PTH model file found")
else:
    pos_test = np.load(os.path.join(WORK_DIR, "positive_features_test.npy"), mmap_mode='r')
    input_shape = pos_test.shape[1:]
    print(f"  Input shape: {input_shape}")
    
    from openwakeword.train import Model
    oww = Model(n_classes=1, input_shape=input_shape, model_type="dnn",
                layer_dim=32, seconds_per_example=1280*input_shape[0]/16000)
    oww.model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    oww.model.eval()
    
    # Run 50 positive and 50 negative examples
    sample_pos = torch.from_numpy(pos_test[:50].astype(np.float32))
    neg_test = np.load(os.path.join(WORK_DIR, "negative_features_test.npy"), mmap_mode='r')
    sample_neg = torch.from_numpy(neg_test[:50].astype(np.float32))
    
    with torch.no_grad():
        pos_preds = oww.model(sample_pos).squeeze().numpy()
        neg_preds = oww.model(sample_neg).squeeze().numpy()
    
    print(f"  Positive predictions: min={pos_preds.min():.4f}, max={pos_preds.max():.4f}, mean={pos_preds.mean():.4f}")
    print(f"  Negative predictions: min={neg_preds.min():.4f}, max={neg_preds.max():.4f}, mean={neg_preds.mean():.4f}")
    print(f"  Pos > 0.5: {(pos_preds > 0.5).sum()}/50, Neg > 0.5: {(neg_preds > 0.5).sum()}/50")
    print(f"  Pos > 0.2: {(pos_preds > 0.2).sum()}/50, Neg > 0.2: {(neg_preds > 0.2).sum()}/50")
    print(f"  Pos > 0.1: {(pos_preds > 0.1).sum()}/50, Neg > 0.1: {(neg_preds > 0.1).sum()}/50")
    print(f"  First 10 positive predictions: {pos_preds[:10].round(4)}")

print("\n" + "=" * 60)
print("3. Check training batch content")
print("=" * 60)
# Check if positive examples look real (non-zero, variable)
pos_train = np.load(os.path.join(WORK_DIR, "positive_features_train.npy"), mmap_mode='r')
print(f"  positive_train global: min={pos_train[:1000].min():.4f}, max={pos_train[:1000].max():.4f}, std={pos_train[:1000].std():.4f}")
# Are there zero examples?
flat = pos_train[:100].reshape(100, -1)
zero_rows = (np.abs(flat).sum(axis=1) < 0.001).sum()
print(f"  Zero/near-zero examples in first 100: {zero_rows}")

print("\nDone.")
