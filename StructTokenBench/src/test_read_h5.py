import h5py
import numpy as np

# Paths to the H5 files
vq_embed_path = 'vq_embed_proteinshake.h5'
vq_embed_apolo_lite_path = 'vq_embed_proteinshake_lite.h5'

# Dataset names (based on typical usage)
dataset_name = '/vq_conformational'  # Adjust if different

def read_and_print_embeddings(file_path, dataset_name, num_rows=10):
    try:
        with h5py.File(file_path, 'r') as f:
            if dataset_name in f:
                data = f[dataset_name][:]
                print(f"Reading from {file_path}, dataset {dataset_name}")
                print(f"Shape: {data.shape}")
                print(f"First {num_rows} rows:")
                for i in range(min(num_rows, data.shape[0])):
                    print(f"Row {i}: {data[i]}")
            else:
                print(f"Dataset {dataset_name} not found in {file_path}")
                print("Available datasets:", list(f.keys()))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Read from vq_embed.h5
read_and_print_embeddings(vq_embed_path, dataset_name)

# Read from vq_embed_apolo_lite.h5
read_and_print_embeddings(vq_embed_apolo_lite_path, dataset_name)


import h5py
p="vq_embed_proteinshake.h5"
with h5py.File(p,"r") as h5:
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print("DATASET apolo lite:", name, obj.shape)
        # uncomment to see groups: else: print("GROUP:", name)
    h5.visititems(walk)


print('---************************************---')

import h5py
p="vq_embed_proteinshake_lite.h5"
with h5py.File(p,"r") as h5:
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print("DATASET large:", name, obj.shape)
        # uncomment to see groups: else: print("GROUP:", name)
    h5.visititems(walk)