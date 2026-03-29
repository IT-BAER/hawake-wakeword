"""Export best_dnn.pth (v1 model) to ONNX with scaler folded into first layer."""
import numpy as np
import torch
import torch.nn as nn
import onnx

# Load scaler stats (computed from same training features)
scaler = np.load('hee_schpustee/scaler_v2.npz')
mean = scaler['mean']
std = scaler['std']
std[std < 1e-8] = 1.0

class WakeWordDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1536, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

model = WakeWordDNN()
model.load_state_dict(torch.load('hee_schpustee/best_dnn.pth', weights_only=False, map_location='cpu'))
model.eval()

# Fold StandardScaler into first linear layer
with torch.no_grad():
    W = model.net[0].weight.numpy().copy()
    b = model.net[0].bias.numpy().copy()
    model.net[0].weight.copy_(torch.from_numpy(W / std[None, :]))
    model.net[0].bias.copy_(torch.from_numpy(b - (W * (mean / std)[None, :]).sum(axis=1)))

class WakeWordONNX(nn.Module):
    def __init__(self, dnn):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dnn = dnn.net
    def forward(self, x): return self.dnn(self.flatten(x))

export_model = WakeWordONNX(model)
export_model.eval()

onnx_path = 'hee_schpustee/hee_schpustee.onnx'
torch.onnx.export(export_model, torch.randn(1, 16, 96), onnx_path,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=11, dynamo=False)

# Fix ONNX for Android compatibility
m = onnx.load(onnx_path)
m.ir_version = 7
for op in m.opset_import:
    if op.domain in ('', 'ai.onnx'): op.version = 11
for node in m.graph.node:
    if node.op_type == 'Reshape':
        for attr in list(node.attribute):
            if attr.name == 'allowzero': node.attribute.remove(attr)
onnx.save(m, onnx_path)
print(f'Exported: {onnx_path}')

# Verify with check_opset
import subprocess
result = subprocess.run(['python', 'check_opset.py', onnx_path], capture_output=True, text=True)
print(result.stdout)
if result.stderr: print(result.stderr)

# Quick verification
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path)
pos = np.load('hee_schpustee/positive_features_test.npy')[:100].astype(np.float32)
ps = sess.run(None, {'input': pos})[0].flatten()
print(f'Positive scores: mean={ps.mean():.4f}, >0.5={float((ps>=0.5).mean()):.4f}, >0.9={float((ps>=0.9).mean()):.4f}')
