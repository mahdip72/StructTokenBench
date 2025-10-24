import h5py
import numpy as np
#
with h5py.File("vq_embed_apolo_lite.h5", "r") as f:
    print("top keys:", list(f.keys()))                 # groups/datasets at root
    d = next((f[k] for k in f.keys() if isinstance(f[k], h5py.Dataset)), None)
    if d is not None:
        print("sample dataset:", d.name, d.shape, d.dtype)
        print("preview:", np.array(d[:3]))




# Inspect InterPro binding lite embeddings
print('--- InterPro binding lite inspection ---')
try:
    p = "vq_embed_interpro_binding_lite.h5"
    with h5py.File(p, "r") as h5:
        # Show top-level keys (groups/datasets)
        try:
            top_keys = list(h5.keys())
        except Exception:
            top_keys = []
        print("ROOT_KEYS (first 50):", top_keys[:50])

        # Count datasets and show a sample
        ds_count = {"n": 0}
        samples = []

        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                ds_count["n"] += 1
                if len(samples) < 20:
                    try:
                        samples.append((name, obj.shape))
                    except Exception:
                        samples.append((name, "<shape unavailable>"))

        h5.visititems(walk)
        print("NUM_DATASETS:", ds_count["n"])
        for name, shape in samples:
            print("DATASET:", name, shape)

        # Probe for specific PDB/chain candidate keys
        def probe_keys(file_obj, pdb_id: str, chain_id: str = "A"):
            chain_up = (chain_id or "").upper()
            bases = [pdb_id, pdb_id.upper()]
            cands = []
            for base in bases:
                cands += [
                    f"{base}_chain_id_{chain_up}",
                    f"{base}_{chain_up}",
                    f"{base}{chain_up}",
                    base,
                ]
            found = []
            for key in cands:
                if key in file_obj:
                    obj = file_obj[key]
                    kind = "Group" if isinstance(obj, h5py.Group) else "Dataset"
                    try:
                        shape = getattr(obj, "shape", None)
                    except Exception:
                        shape = None
                    found.append((key, kind, shape))
            return cands, found

        for pdb in ["5hoa", "5u6c", "5uad"]:
            cands, found = probe_keys(h5, pdb)
            print(f"CANDIDATES for {pdb}_A:", cands)
            print(f"FOUND for {pdb}_A:", found)
except FileNotFoundError:
    print("File not found: vq_embed_interpro_binding_lite.h5 (place it next to this script to inspect)")
except Exception as e:
    print("Error inspecting InterPro binding lite H5:", repr(e))



def show(path):
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"[G] {name}/")
            elif isinstance(obj, h5py.Dataset):
                print(f"[D] {name}  shape={obj.shape} dtype={obj.dtype}")
        f.visititems(visit)

        # quick hint: print attributes on the file and first dataset
        if len(f.attrs):
            print("\n[file attrs]", dict(f.attrs))
        for k in f.keys():
            if isinstance(f[k], h5py.Dataset):
                ds = f[k]
                print(f"\npreview of {ds.name}:\n", ds[:3])  # first 3 rows/elements
                break

show("vq_embed_apolo_lite.h5")