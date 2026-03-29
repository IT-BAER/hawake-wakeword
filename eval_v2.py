import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession('hee_schpustee/hee_schpustee.onnx')
acav = np.load('openwakeword_features_ACAV100M_2000_hrs_16bit.npy', mmap_mode='r')

n_check = 500000
batch = 50000
all_scores = []
print(f'Scoring {n_check} ACAV (bulk slices)...', flush=True)
for s in range(0, n_check, batch):
    e = min(s + batch, n_check)
    chunk = np.ascontiguousarray(acav[s:e]).astype(np.float32)
    scores = sess.run(None, {'input': chunk})[0].flatten()
    all_scores.append(scores)
    print(f'  {e}/{n_check}', flush=True)

all_scores = np.concatenate(all_scores)
hrs = n_check * 1.28 / 3600
print(f'\nFP/hr over {hrs:.0f} hrs:', flush=True)
for t in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 0.9999]:
    fps = int((all_scores >= t).sum())
    print(f'  t={t}: {fps} FPs = {fps/hrs:.1f}/hr')

p = np.percentile(all_scores, [99, 99.5, 99.9, 99.99])
print(f'p99={p[0]:.4f} p99.5={p[1]:.4f} p99.9={p[2]:.4f} p99.99={p[3]:.4f}')

pos = np.load('hee_schpustee/positive_features_test.npy')[:5000].astype(np.float32)
ps = sess.run(None, {'input': pos})[0].flatten()
print(f'\nPos recall: >0.5={float((ps>=0.5).mean()):.4f} >0.9={float((ps>=0.9).mean()):.4f} >0.99={float((ps>=0.99).mean()):.4f}')
