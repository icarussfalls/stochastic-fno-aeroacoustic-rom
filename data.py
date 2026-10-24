import h5py
import numpy as np

# Open the file in read-only mode
with h5py.File('spod_matlab/spod_complete_database.mat', 'r') as f:
    print('\n--- FILE CONTENTS ---')
    st_vector = np.array(f['St_vector']).squeeze()
    for k, v in f.items():
        if hasattr(v, 'shape'):
            print(f'Variable: {k} | Shape: {v.shape} | Type: {v.dtype}')



print(f"\n--- DETECTED STROUHAL NUMBERS (Total: {len(st_vector)}) ---")
for idx, st_val in enumerate(st_vector):
    print(f"Index {idx:03d} | St = {st_val:.4f}")