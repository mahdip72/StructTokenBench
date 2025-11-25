from __future__ import annotations
import os
import h5py
import numpy as np
import torch

from src.protein_chain import WrappedProteinChain as PC


class WrappedMyRepAtlasTokenizer:
    """
    Continuous-embedding tokenizer for ATLAS physicochemical tasks.

    - Reads per-residue embeddings from an H5 file organized as groups per entry
      (e.g., 1abc, 1abc_chain_id_A), each containing datasets like 'embedding' and
      optionally 'indices'.
    - Aligns output residue_index to mmCIF/PDB author numbering when indices are available,
      otherwise uses residue indices parsed from structure.
    - Robust to missing entries: synthesizes zeros of length L when needed.
    """

    pad_token_id = 0

    def __init__(
        self,
        h5_path: str | None = None,
        embeddings_dataset: str | None = None,
        embed_dim: int = 128,
        fallback_to_any_chain: bool = True,
        device: str | None = None,
        **kwargs,
    ):
        self.device = device or "cpu"
        self.embed_dim = int(embed_dim)

        self.h5_path = None
        self.h5 = None
        self.emb = None
        self._last_debug_info = None
        self._last_h5_key = None
        # lazily-built index of entry name -> relative group path containing 'embedding'
        self._entry_index = None

        if h5_path:
            self.h5_path = os.path.abspath(os.path.expanduser(h5_path))
            if not os.path.isfile(self.h5_path):
                raise FileNotFoundError(f"H5 not found: {self.h5_path}")
            self.h5 = h5py.File(self.h5_path, "r")
            # If a specific subgroup is provided, try that first; otherwise root
            if isinstance(embeddings_dataset, str) and len(embeddings_dataset.strip()) > 0:
                key = embeddings_dataset.lstrip("/")
                if key in self.h5:
                    self.emb = self.h5[key]
                else:
                    # Allow direct group path
                    grp = self.h5.get(embeddings_dataset)
                    if isinstance(grp, (h5py.Group, h5py.Dataset)):
                        self.emb = grp
                    else:
                        self.emb = None
            else:
                self.emb = self.h5  # search from root by default

            # Try to infer embed_dim from a sample
            try:
                sample = None
                if isinstance(self.emb, h5py.Group):
                    for k in self.emb.keys():
                        obj = self.emb[k]
                        if isinstance(obj, h5py.Group) and "embedding" in obj:
                            sample = obj["embedding"][()]
                            break
                        if isinstance(obj, h5py.Dataset):
                            sample = obj[()]
                            break
                elif isinstance(self.emb, h5py.Dataset):
                    sample = self.emb[()]
                if sample is not None:
                    if getattr(sample, "ndim", 0) == 1:
                        sample = sample[None, :]
                    self.embed_dim = int(sample.shape[-1])
            except Exception:
                pass

        self.fallback = bool(fallback_to_any_chain)

    def get_num_tokens(self):
        # Signal continuous features
        return None

    def _candidate_keys(self, pdb_id: str, chain_up: str):
        bases = [pdb_id, pdb_id.upper()]
        cands = []
        for base in bases:
            if chain_up:
                cands += [
                    f"{base}_chain_id_{chain_up}",
                    f"{base}_{chain_up}",
                    f"{base}{chain_up}",
                ]
            cands.append(base)
        return cands

    def _read_from_h5(self, pdb_id: str, chain_up: str):
        """
        Return (embedding_array, indices_array) if found.
        Expects groups with 'embedding' and optionally 'indices' datasets.
        """
        grp = self.emb if self.emb is not None else self.h5
        try:
            if isinstance(grp, h5py.Group):
                for k in self._candidate_keys(pdb_id, chain_up):
                    if k in grp:
                        obj = grp[k]
                        if isinstance(obj, h5py.Group):
                            arr = None
                            idx = None
                            if "embedding" in obj and isinstance(obj["embedding"], h5py.Dataset):
                                self._last_h5_key = f"{k}/embedding"
                                arr = obj["embedding"][()]
                            if "indices" in obj and isinstance(obj["indices"], h5py.Dataset):
                                idx = obj["indices"][()]
                            if arr is not None:
                                return arr, idx
                        if isinstance(obj, h5py.Dataset):
                            self._last_h5_key = k
                            return obj[()], None
                # If immediate children don't match, try a lazy recursive index
                if self.fallback:
                    if self._entry_index is None:
                        # Build a map from plausible entry keys -> relative group path
                        index = {}
                        def visit(name, obj):
                            if isinstance(obj, h5py.Group):
                                if "embedding" in obj:
                                    leaf = name.split("/")[-1]
                                    # index both full leaf and 4-char base id variants
                                    keys = {leaf, leaf.lower(), leaf.upper()}
                                    if len(leaf) >= 4:
                                        base = leaf[:4]
                                        keys.update({base, base.lower(), base.upper()})
                                    for kk in keys:
                                        if kk not in index:
                                            index[kk] = name
                        grp.visititems(visit)
                        self._entry_index = index
                    # resolve via candidate keys against the index
                    for cand in self._candidate_keys(pdb_id, chain_up):
                        for probe in (cand, cand.lower(), cand.upper()):
                            path = self._entry_index.get(probe, None)
                            if path is None:
                                continue
                            obj = grp[path]
                            if isinstance(obj, h5py.Group):
                                arr = None
                                idx = None
                                if "embedding" in obj and isinstance(obj["embedding"], h5py.Dataset):
                                    self._last_h5_key = f"{path}/embedding"
                                    arr = obj["embedding"][()]
                                if "indices" in obj and isinstance(obj["indices"], h5py.Dataset):
                                    idx = obj["indices"][()]
                                if arr is not None:
                                    return arr, idx
                            if isinstance(obj, h5py.Dataset):
                                self._last_h5_key = path
                                return obj[()], None
                if self.fallback:
                    # Pick first dataset/group if specific key not found
                    for k in grp.keys():
                        obj = grp[k]
                        if isinstance(obj, h5py.Group):
                            if "embedding" in obj:
                                self._last_h5_key = f"{k}/embedding"
                                arr = obj["embedding"][()]
                                idx = obj["indices"][()] if "indices" in obj else None
                                return arr, idx
                        if isinstance(obj, h5py.Dataset):
                            self._last_h5_key = k
                            return obj[()], None
            elif isinstance(grp, h5py.Dataset):
                self._last_h5_key = "/"
                return grp[()], None
        except Exception:
            pass
        return None, None

    @torch.no_grad()
    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence: bool = False):
        pdb_id = os.path.basename(pdb_path).split(".")[0].lower()
        user_chain = (chain_id or "").strip().upper()

        # 1) Parse structure FIRST with 'detect' to get author residue indices and inferred chain
        resid = None
        seqs_real = None
        pc_len = None
        inferred_chain = None
        try:
            if str(pdb_path).lower().endswith(".pdb"):
                pc = PC.from_pdb(pdb_path, "detect", id=pdb_id)
            else:
                pc = PC.from_cif(pdb_path, "detect", id=pdb_id)
            resid = np.asarray(pc.residue_index, dtype=int)
            seqs_real = list(pc.sequence)
            pc_len = int(resid.shape[0]) if resid is not None else None
            try:
                inferred_chain = getattr(pc, "chain_id", None)
                if isinstance(inferred_chain, str):
                    inferred_chain = inferred_chain.strip().upper()
            except Exception:
                inferred_chain = None
        except Exception:
            resid = None
            seqs_real = None
            pc_len = None
            inferred_chain = None

        # 2) Query H5 using the best available chain key
        chain_for_h5 = user_chain or inferred_chain or ""
        arr, idx = self._read_from_h5(pdb_id, chain_for_h5)
        raw_len = None
        raw_idx_len = None
        key_used = self._last_h5_key
        if arr is not None and getattr(arr, "ndim", 0) == 1:
            arr = arr[None, :]
        if arr is not None:
            raw_len = int(arr.shape[0])

        # 3) If H5 missing, synthesize zeros matching structure length (if available)
        if arr is None:
            L = int(len(resid)) if resid is not None else 128
            arr = np.zeros((L, self.embed_dim), dtype=np.float32)

        # IMPORTANT: produce CPU tensor here to avoid CUDA init in DataLoader workers
        feats = torch.as_tensor(arr, dtype=torch.float32)
        L = int(feats.shape[0])

        # Prefer indices from H5 when available and matching length; otherwise use author indices
        used_idx_from_h5 = False
        if idx is not None:
            try:
                idx = np.asarray(idx).astype(int)
                raw_idx_len = int(idx.shape[0])
            except Exception:
                idx = None
        if idx is not None and idx.shape[0] == L:
            resid = idx
            used_idx_from_h5 = True
        elif resid is None or resid.shape[0] != L:
            resid = np.arange(L, dtype=int)
        used_pc_resid = (resid is not None) and (not used_idx_from_h5)

        seqs = seqs_real if (use_sequence and seqs_real is not None and len(seqs_real) == L) else ["X"] * L

        # Record debug info for upstream alignment warnings
        self._last_debug_info = {
            "pdb_id": pdb_id,
            "chain_id": chain_for_h5,
            "h5_key": key_used,
            "h5_raw_len": raw_len,
            "h5_idx_raw_len": raw_idx_len,
            "pc_resid_len": pc_len,
            "final_L": L,
            "used_idx_from_h5": used_idx_from_h5,
            "used_pc_resid": used_pc_resid,
        }

        return feats, resid, seqs

    def get_last_debug_info(self):
        return self._last_debug_info

    def close(self):
        try:
            if self.h5 is not None:
                self.h5.close()
        except Exception:
            pass
