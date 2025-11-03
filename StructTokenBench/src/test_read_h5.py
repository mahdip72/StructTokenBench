# h5_features.py  (parser-free)
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import h5py
import numpy as np

from protein_chain import WrappedProteinChain as PC

__all__ = [
    "EMBED_NAME",
    "INDEX_NAME",
    "LoadResult",
    "H5FeatureLoader",
    "rewrite_with_author_indices",
]

EMBED_NAME = "embedding"
INDEX_NAME = "indices"


def _copy_attrs(src, dst):
    for k, v in src.attrs.items():
        try:
            dst.attrs[k] = v
        except Exception:
            # silently skip non-copyable attrs
            pass


def _resolve_entry(h5: h5py.File, base_id: str, chain_id: Optional[str] = None) -> str:
    """
    Try common variations like 1a2x, 1A2X, 1a2x_chain_id_A, 1a2x_A, 1a2xA.
    Returns the first matching group name or raises KeyError.
    """
    cands: List[str] = []
    for b in (base_id, base_id.lower(), base_id.upper()):
        cands.append(b)
        if chain_id:
            c = chain_id.upper()
            cands += [f"{b}_chain_id_{c}", f"{b}_{c}", f"{b}{c}"]
    for name in cands:
        if name in h5:
            return name
    raise KeyError(f"Could not resolve entry for {base_id!r} (chain={chain_id!r}). Tried: {cands}")


def _mask_drop_value(idx: np.ndarray, drop_value: int = -1) -> np.ndarray:
    if idx.ndim != 1:
        raise ValueError(f"indices must be 1D; got {idx.shape}")
    if not np.issubdtype(idx.dtype, np.integer):
        raise TypeError(f"indices must be int; got {idx.dtype}")
    return (idx != drop_value)


@dataclass
class LoadResult:
    entry_path: str
    embedding: Optional[np.ndarray]
    indices: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    dropped_rows: int
    original_rows: Optional[int]
    info: Dict[str, Any]


class H5FeatureLoader:
    """
    Importable loader/cleaner for (embedding, indices) pairs in your HDF5.
    No command-line interface; call these methods directly.

    Example:
        loader = H5FeatureLoader("features.h5")
        print(loader.list_entries()[:5])

        r = loader.load("1a2x", chain_id="A", clean=True, drop_value=-1, return_mask=True)
        print(r.info)

        # write a cleaned copy (removing rows where indices == -1)
        report = loader.delete_neg1(dst_path="features.cleaned.h5", inplace=False)
        print(report[:3])
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path

    # --------- Discovery & loading ---------

    def list_entries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with h5py.File(self.h5_path, "r") as h5:
            def visit(name, obj):
                if isinstance(obj, h5py.Group):
                    has_emb = EMBED_NAME in obj.keys()
                    has_idx = INDEX_NAME in obj.keys()
                    if has_emb or has_idx:
                        out.append({"entry": name, "has_embedding": has_emb, "has_indices": has_idx})

            h5.visititems(visit)
        return out[:limit] if limit else out

    def load(
        self,
        entry: str,
        chain_id: Optional[str] = None,
        clean: bool = True,
        drop_value: int = -1,
        return_mask: bool = False,
    ) -> LoadResult:
        """
        Load embedding/indices by entry (+ optional chain).
        If clean=True, drop rows where indices == drop_value from BOTH arrays.
        """
        with h5py.File(self.h5_path, "r") as h5:
            grp_name = _resolve_entry(h5, entry, chain_id)
            grp = h5[grp_name]
            emb = np.array(grp[EMBED_NAME]) if EMBED_NAME in grp else None
            idx = np.array(grp[INDEX_NAME]) if INDEX_NAME in grp else None

            info = {
                "entry_path": grp_name,
                "has_embedding": emb is not None,
                "has_indices": idx is not None,
                "embedding_shape": None if emb is None else tuple(emb.shape),
                "indices_shape": None if idx is None else tuple(idx.shape),
                "embedding_dtype": None if emb is None else str(emb.dtype),
                "indices_dtype": None if idx is None else str(idx.dtype),
            }

            mask = None
            dropped = 0
            orig_n = None
            if clean and idx is not None:
                mask = _mask_drop_value(idx, drop_value=drop_value)
                orig_n = idx.shape[0]
                dropped = int((~mask).sum())
                idx = idx[mask]
                if emb is not None:
                    if emb.shape[0] != orig_n:
                        raise ValueError(
                            f"Embedding first dim {emb.shape[0]} != indices length {orig_n} in {grp_name}"
                        )
                    emb = emb[mask, ...]

            return LoadResult(
                entry_path=grp_name,
                embedding=emb,
                indices=idx,
                mask=mask if return_mask else None,
                dropped_rows=dropped,
                original_rows=orig_n,
                info=info,
            )

    def overview(self, sample: int = 20) -> List[Dict[str, Any]]:
        """Quick per-entry summary (shapes, dtypes, count of -1 in indices)."""
        report: List[Dict[str, Any]] = []
        with h5py.File(self.h5_path, "r") as h5:
            def visit(name, obj):
                if isinstance(obj, h5py.Group):
                    has_emb = EMBED_NAME in obj.keys()
                    has_idx = INDEX_NAME in obj.keys()
                    if not (has_emb or has_idx):
                        return
                    row: Dict[str, Any] = {"entry": name, "has_embedding": has_emb, "has_indices": has_idx}
                    try:
                        if has_emb:
                            e = obj[EMBED_NAME]
                            row["embedding_shape"] = tuple(e.shape)
                            row["embedding_dtype"] = str(e.dtype)
                        if has_idx:
                            x = obj[INDEX_NAME][...]
                            row["indices_shape"] = tuple(x.shape)
                            row["indices_dtype"] = str(x.dtype)
                            row["neg1_count"] = int((x == -1).sum()) if x.ndim == 1 else None
                    except Exception as e:
                        row["error"] = repr(e)
                    report.append(row)

            h5.visititems(visit)
        return report[:sample] if sample else report

    # --------- Cleaning utilities (no CLI) ---------

    def delete_neg1(
        self,
        dst_path: Optional[str] = None,
        entries: Optional[List[Tuple[str, Optional[str]]]] = None,
        drop_value: int = -1,
        inplace: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Remove rows where indices == drop_value from (indices, embedding).
        - If entries is None: process ALL groups containing indices.
        - Writes to dst_path unless inplace=True (then edits self.h5_path via safe replace).
        Returns a per-group report.
        """
        src_path = self.h5_path
        if inplace:
            tmp = src_path + ".tmp.cleaning"
            report = self._clean_to(src_path, tmp, entries, drop_value)
            os.replace(tmp, src_path)
            return report
        else:
            if dst_path is None:
                root, ext = os.path.splitext(src_path)
                dst_path = root + ".cleaned" + ext
            return self._clean_to(src_path, dst_path, entries, drop_value)

    def _clean_to(
        self,
        src_path: str,
        dst_path: str,
        entries: Optional[List[Tuple[str, Optional[str]]]],
        drop_value: int,
    ) -> List[Dict[str, Any]]:
        # Resolve explicit targets (if any)
        targets: Optional[set] = None
        with h5py.File(src_path, "r") as src:
            if entries:
                names = set()
                for (e, c) in entries:
                    names.add(_resolve_entry(src, e, c))
                targets = names

        report: List[Dict[str, Any]] = []

        with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
            _copy_attrs(src, dst)

            def process_group(src_grp: h5py.Group, dst_grp: h5py.Group):
                path_name = src_grp.name
                must_clean = (targets is None) or (path_name in targets)
                has_idx = isinstance(src_grp.get(INDEX_NAME, None), h5py.Dataset)
                has_emb = isinstance(src_grp.get(EMBED_NAME, None), h5py.Dataset)

                mask = None
                orig_n = None
                removed = 0
                if must_clean and has_idx:
                    idx_ds = src_grp[INDEX_NAME]
                    if idx_ds.ndim == 1 and np.issubdtype(idx_ds.dtype, np.integer):
                        idx = idx_ds[...]
                        orig_n = idx.shape[0]
                        mask = (idx != drop_value)
                        removed = int((~mask).sum())

                for name, obj in src_grp.items():
                    if isinstance(obj, h5py.Group):
                        new_sub = dst_grp.create_group(name)
                        _copy_attrs(obj, new_sub)
                        process_group(obj, new_sub)
                    elif isinstance(obj, h5py.Dataset):
                        data = obj[...]
                        if mask is not None:
                            if name == INDEX_NAME and data.ndim == 1 and data.shape[0] == mask.shape[0]:
                                data = data[mask]
                            elif name == EMBED_NAME and data.shape[0] == mask.shape[0]:
                                data = data[mask, ...]
                        kwargs = {}
                        if obj.compression is not None:
                            kwargs["compression"] = obj.compression
                        if obj.chunks is not None:
                            # keep original chunking when feasible
                            kwargs["chunks"] = obj.chunks
                        dnew = dst_grp.create_dataset(name, data=data, dtype=obj.dtype, **kwargs)
                        _copy_attrs(obj, dnew)

                if mask is not None and orig_n is not None:
                    report.append({
                        "group_path": path_name,
                        "had_embedding": has_emb,
                        "had_indices": has_idx,
                        "original_rows": int(orig_n),
                        "removed_rows": int(removed),
                        "kept_rows": int(mask.sum()),
                    })

            # copy root
            for name, obj in src.items():
                if isinstance(obj, h5py.Group):
                    new_grp = dst.create_group(name)
                    _copy_attrs(obj, new_grp)
                    process_group(obj, new_grp)
                elif isinstance(obj, h5py.Dataset):
                    data = obj[...]
                    kwargs = {}
                    if obj.compression is not None:
                        kwargs["compression"] = obj.compression
                    if obj.chunks is not None:
                        kwargs["chunks"] = obj.chunks
                    dnew = dst.create_dataset(name, data=data, dtype=obj.dtype, **kwargs)
                    _copy_attrs(obj, dnew)

        return report


# ---------- Author index rewrite pipeline (standalone function) ----------

def _parse_group_name_for_ids(group_name: str) -> Tuple[str, Optional[str]]:
    """
    Extract (pdb_id, chain_id) from common H5 group name patterns like:
      - 1a2x
      - 1a2x_chain_id_A
      - 1a2x_A
      - 1a2xA
    Returns (pdb_id, chain_id or None).
    """
    base = group_name.strip().split("/")[-1]
    m = re.match(r"^([0-9][A-Za-z0-9]{3})_chain_id_([A-Za-z0-9])$", base)
    if m:
        return m.group(1).lower(), m.group(2).upper()
    m = re.match(r"^([0-9][A-Za-z0-9]{3})_([A-Za-z0-9])$", base)
    if m:
        return m.group(1).lower(), m.group(2).upper()
    m = re.match(r"^([0-9][A-Za-z0-9]{3})([A-Za-z0-9])$", base)
    if m:
        return m.group(1).lower(), m.group(2).upper()
    m = re.match(r"^([0-9][A-Za-z0-9]{3})$", base)
    if m:
        return m.group(1).lower(), None
    # fallback: use first 4 as pdb, leave chain None
    return base[:4].lower(), None


def _find_mmcif_path(pdb_data_dir: str, pdb_id: str) -> Optional[str]:
    """Locate mmCIF file path for a given pdb_id (a few common layouts)."""
    cands = [
        os.path.join(pdb_data_dir, "mmcif_files", "mmcif_files", f"{pdb_id}.cif"),
        os.path.join(pdb_data_dir, "mmcif_files", f"{pdb_id}.cif"),
        os.path.join(pdb_data_dir, f"{pdb_id}.cif"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def rewrite_with_author_indices(
    src_h5_path: str,
    dst_h5_path: str,
    pdb_data_dir: str,
    atol_zero: float = 1e-8,
) -> List[Dict[str, Any]]:
    """
    Create a corrected H5:
      - drop rows with indices == -1 (if present)
      - drop rows whose embedding rows are all zeros (within atol)
      - replace/write indices with mmCIF author residue indices for that PDB/chain
        (paired by order up to min(len(embedding_clean), len(residue_index))).
    Returns a per-group report summarizing changes.
    """
    report: List[Dict[str, Any]] = []
    with h5py.File(src_h5_path, "r") as src, h5py.File(dst_h5_path, "w") as dst:
        _copy_attrs(src, dst)

        def process_group(src_grp: h5py.Group, dst_grp: h5py.Group):
            has_emb = EMBED_NAME in src_grp
            has_idx = INDEX_NAME in src_grp

            # Recurse first for subgroups
            for name, obj in src_grp.items():
                if isinstance(obj, h5py.Group):
                    new_sub = dst_grp.create_group(name)
                    _copy_attrs(obj, new_sub)
                    process_group(obj, new_sub)

            if not has_emb:
                return  # nothing to fix at this level

            # Work at this group level if it directly contains datasets
            try:
                emb = np.array(src_grp[EMBED_NAME])
                idx = np.array(src_grp[INDEX_NAME]) if has_idx else None
            except Exception:
                return

            grp_name = src_grp.name
            pdb_id, chain_id = _parse_group_name_for_ids(grp_name)
            mmcif = _find_mmcif_path(pdb_data_dir, pdb_id)
            resid = None
            if mmcif is not None:
                try:
                    pc = PC.from_cif(mmcif, (chain_id or "detect"), id=pdb_id)
                    resid = np.asarray(pc.residue_index, dtype=int)
                except Exception:
                    resid = None

            # Clean rows
            if emb.ndim == 1:
                emb = emb[None, :]
            mask_zero = ~np.all(np.isclose(emb, 0.0, atol=atol_zero), axis=-1)
            mask_idx = np.ones((emb.shape[0],), dtype=bool)
            if idx is not None and idx.ndim == 1 and idx.shape[0] == emb.shape[0]:
                mask_idx = (idx != -1)
            mask = mask_zero & mask_idx
            emb_clean = emb[mask]

            # Author indices: pair in order up to min length
            if resid is not None and resid.size > 0:
                L = int(min(emb_clean.shape[0], resid.shape[0]))
                emb_out = emb_clean[:L]
                idx_out = resid[:L]
            else:
                # No structure available; keep cleaned embeddings and fallback to 0..L-1
                emb_out = emb_clean
                idx_out = np.arange(emb_out.shape[0], dtype=int)

            # Write current group's datasets (overwriting if present)
            # Remove existing datasets if any
            for name in list(dst_grp.keys()):
                try:
                    del dst_grp[name]
                except Exception:
                    pass
            d_e = dst_grp.create_dataset(EMBED_NAME, data=emb_out, dtype=emb_out.dtype)
            d_i = dst_grp.create_dataset(INDEX_NAME, data=idx_out, dtype=idx_out.dtype)
            _copy_attrs(src_grp[EMBED_NAME], d_e)
            if INDEX_NAME in src_grp:
                _copy_attrs(src_grp[INDEX_NAME], d_i)

            report.append({
                "group_path": grp_name,
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "orig_rows": int(emb.shape[0]),
                "kept_rows": int(emb_out.shape[0]),
                "dropped_rows": int(emb.shape[0] - emb_clean.shape[0]),
                "used_author_indices": bool(resid is not None),
            })

        # Start from root
        for name, obj in src.items():
            if isinstance(obj, h5py.Group):
                new_grp = dst.create_group(name)
                _copy_attrs(obj, new_grp)
                process_group(obj, new_grp)
            elif isinstance(obj, h5py.Dataset):
                data = obj[...]
                dnew = dst.create_dataset(name, data=data, dtype=obj.dtype)
                _copy_attrs(obj, dnew)

    return report


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="H5 feature utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_over = sub.add_parser("overview", help="List entries with shapes and -1 counts")
    p_over.add_argument("h5", type=str)
    p_over.add_argument("--sample", type=int, default=10)
    p_clean = sub.add_parser("delete_neg1", help="Write copy with rows idx==-1 removed")
    p_clean.add_argument("h5", type=str)
    p_clean.add_argument("--out", type=str, default=None)
    p_clean.add_argument("--inplace", action="store_true")
    p_auth = sub.add_parser("rewrite_author",
                            help="Write copy with rows cleaned and indices rewritten to mmCIF author numbering")
    p_auth.add_argument("h5", type=str)
    p_auth.add_argument("pdb_data_dir", type=str, help="Path to PDB data directory containing mmcif_files/")
    p_auth.add_argument("--out", type=str, default=None)

    args = parser.parse_args()

    if args.cmd == "overview":
      print(list_entries(args.h5)[: args.sample])

    elif args.cmd == "delete_neg1":
        loader = H5FeatureLoader(args.h5)
        rep = loader.delete_neg1(dst_path=args.out, inplace=args.inplace)
        print(rep[:10])

    elif args.cmd == "rewrite_author":
        src = args.h5
        out = args.out or (os.path.splitext(src)[0] + ".author.h5")
        rep = rewrite_with_author_indices(src, out, args.pdb_data_dir)
        print(f"Wrote {out}; sample report:", rep[:10])
