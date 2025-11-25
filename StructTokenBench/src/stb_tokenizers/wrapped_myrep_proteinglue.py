# stb_tokenizers/wrapped_myrep_proteinglue.py
import os
import h5py
import numpy as np
import torch

from src.protein_chain import WrappedProteinChain as PC


class WrappedMyRepProteinGLUETokenizer:
    """
    Continuous-embedding tokenizer for ProteinGLUE tasks (e.g., EpitopeRegion).

    - Optionally reads per-residue embeddings from an H5 dataset (default: /vq_embed_proteinglue)
    - Keys are tried in several forms, e.g. "<pdb_id>_chain_id_<CHAIN>", "<pdb_id>_<CHAIN>", "<pdb_id><CHAIN>", "<pdb_id>"
    - Returns a float tensor [L, D] for each chain, with residue_index aligned to mmCIF author numbering when possible.

    Design goal: be runnable even without an H5 file; if no H5 is provided or a key is missing,
    the tokenizer synthesizes a zero feature matrix of shape [L, D] using the chain length from the mmCIF.
    """

    pad_token_id = 0

    def __init__(
        self,
        h5_path: str | None = None,
        embeddings_dataset: str = "/vq_embed_proteinglue",
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

        if h5_path:
            self.h5_path = os.path.abspath(os.path.expanduser(h5_path))
            if not os.path.isfile(self.h5_path):
                raise FileNotFoundError(f"H5 not found: {self.h5_path}")
            self.h5 = h5py.File(self.h5_path, "r")
            ds = embeddings_dataset.lstrip("/") if isinstance(embeddings_dataset, str) else ""
            if ds in self.h5:
                self.emb = self.h5[ds]
            else:
                # Allow pointing directly at a group (no leading slash) or the root
                if isinstance(self.h5.get(embeddings_dataset), (h5py.Group, h5py.Dataset)):
                    self.emb = self.h5[embeddings_dataset]
                else:
                    self.emb = None

            # Try to infer embed_dim from the first item if possible
            try:
                if isinstance(self.emb, h5py.Group):
                    first_key = next(iter(self.emb.keys()))
                    sample = self.emb[first_key][()]
                elif isinstance(self.emb, h5py.Dataset):
                    sample = self.emb[()]
                else:
                    sample = None
                if sample is not None:
                    if sample.ndim == 1:
                        sample = sample[None, :]
                    self.embed_dim = int(sample.shape[-1])
            except Exception:
                pass

        self.fallback = bool(fallback_to_any_chain)

    # expected by the runner to detect continuous features
    def get_num_tokens(self):
        return None

    def _candidate_keys(self, pdb_id: str, chain_up: str):
        """Generate candidate H5 keys for both lower/upper PDB ids.
        Examples per (pdb='1a0h', chain='A'):
          - 1a0h_chain_id_A, 1a0h_A, 1a0hA, 1a0h
          - 1A0H_chain_id_A, 1A0H_A, 1A0HA, 1A0H
        """
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
        - embedding_array: np.ndarray (L, D) or (L,)
        - indices_array:   np.ndarray (L,) of residue indices if available, else None
        """
        if self.emb is None:
            return None, None
        try:
            if isinstance(self.emb, h5py.Group):
                for k in self._candidate_keys(pdb_id, chain_up):
                    if k in self.emb:
                        obj = self.emb[k]
                        if isinstance(obj, h5py.Dataset):
                            return obj[()], None
                        if isinstance(obj, h5py.Group):
                            arr = None
                            idx = None
                            # prefer common embedding dataset names
                            for sub in ("embedding", "embeddings", "features"):
                                if sub in obj and isinstance(obj[sub], h5py.Dataset):
                                    arr = obj[sub][()]
                                    break
                            # optional indices
                            if "indices" in obj and isinstance(obj["indices"], h5py.Dataset):
                                idx = obj["indices"][()]
                            if arr is not None:
                                return arr, idx
                if self.fallback:
                    # fall back to any entry: pick first dataset within the group
                    for k in self.emb.keys():
                        obj = self.emb[k]
                        if isinstance(obj, h5py.Dataset):
                            return obj[()], None
                        if isinstance(obj, h5py.Group):
                            arr = None
                            idx = None
                            for sub in ("embedding", "embeddings", "features"):
                                if sub in obj and isinstance(obj[sub], h5py.Dataset):
                                    arr = obj[sub][()]
                                    break
                            if "indices" in obj and isinstance(obj["indices"], h5py.Dataset):
                                idx = obj["indices"][()]
                            if arr is not None:
                                return arr, idx
            elif isinstance(self.emb, h5py.Dataset):
                # A single dataset for this entire H5 (rare). Use as-is.
                return self.emb[()], None
        except Exception:
            pass
        return None, None

    @torch.no_grad()
    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence: bool = False):
        """
        Return per-residue continuous features aligned to one chain.
        Outputs:
          token_ids:     (L, D) float32 tensor
          residue_index: (L,) int numpy array (author numbering if available, otherwise 0..L-1)
          seqs:          list[str] length L (dummy or real sequence)
        """
        pdb_id = os.path.basename(pdb_path).split(".")[0].lower()
        chain_up = (chain_id or "").strip().upper()

        # 1) Try to fetch features from H5 if available
        arr, idx = self._read_from_h5(pdb_id, chain_up)
        # If we have H5 arrays, drop rows with idx == -1 and rows whose embeddings are all zeros
        if arr is not None:
            if arr.ndim == 1:
                arr = arr[None, :]
            # Normalize idx to int array if present
            if idx is not None:
                try:
                    idx = np.asarray(idx).astype(int)
                except Exception:
                    idx = None
            try:
                row_nonzero = ~np.all(arr == 0, axis=-1)
            except Exception:
                row_nonzero = np.ones((arr.shape[0],), dtype=bool)
            if idx is not None and hasattr(idx, "shape") and idx.shape[0] == arr.shape[0]:
                mask = (idx != -1) & row_nonzero
            else:
                mask = row_nonzero
            try:
                arr = arr[mask]
                if idx is not None and hasattr(idx, "shape") and idx.shape[0] == row_nonzero.shape[0]:
                    idx = idx[mask]
            except Exception:
                pass

        # 2) Load structure to determine length and residue numbering
        try:
            pc = PC.from_cif(pdb_path, chain_up or "detect", id=pdb_id)
            resid = np.asarray(pc.residue_index, dtype=int)
            seqs_real = list(pc.sequence)
        except Exception:
            # If parsing fails, use a best-effort fallback
            resid = None
            seqs_real = None

        if arr is None:
            # Graceful fallback: zeros of length L with configured embed_dim
            if resid is None:
                L = 128
            else:
                L = int(len(resid))
            arr = np.zeros((L, self.embed_dim), dtype=np.float32)

        if arr.ndim == 1:
            arr = arr[None, :]
        feats = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        L = int(feats.shape[0])

        # Prefer indices provided by H5 group when available and matching length
        if idx is not None:
            try:
                idx = np.asarray(idx).astype(int)
            except Exception:
                idx = None
        if idx is not None and idx.shape[0] == L:
            resid = idx
        elif resid is None or resid.shape[0] != L:
            resid = np.arange(L, dtype=int)
        seqs = seqs_real if (use_sequence and seqs_real is not None and len(seqs_real) == L) else ["X"] * L

        return feats, resid, seqs

    def close(self):
        try:
            if self.h5 is not None:
                self.h5.close()
        except Exception:
            pass


