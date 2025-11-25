
# stb_debug_utils.py
# Drop-in utilities to debug NaN correlations & constant predictions in Lightning.
# Usage (inside your LightningModule):
#   from stb_debug_utils import log_batch_debug
#   ...
#   def validation_step(...):
#       y_pred, y_true = ... # tensors, same shape
#       mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
#       log_batch_debug(self, y_true, y_pred, mask=mask, prefix="validation")
#
# Works with CPU/GPU tensors; converts to numpy safely.

import numpy as np
from typing import Optional

def _to_numpy(x):
    import torch
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    # fallback
    return np.asarray(x, dtype=np.float32)

def safe_corr(y_true, y_pred, mask: Optional[np.ndarray]=None, method: str="pearson"):

    y_true = _to_numpy(y_true).ravel()
    y_pred = _to_numpy(y_pred).ravel()
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        mask = _to_numpy(mask).astype(bool).ravel()
        valid = valid & mask

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    n = y_true.shape[0]
    if n < 3:
        return np.nan

    if method.lower().startswith("spear"):
        # rank transform with average ranks for ties
        y_true = _rankdata(y_true)
        y_pred = _rankdata(y_pred)

    # de-mean
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()

    denom = (np.sqrt((yt**2).sum()) * np.sqrt((yp**2).sum()))
    if denom == 0.0 or not np.isfinite(denom):
        return np.nan
    return float((yt * yp).sum() / denom)

def _rankdata(a):
    # Simple average-rank implementation to avoid SciPy dependency
    a = np.asarray(a)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense_rank = obs.cumsum() - 1
    # average ranks per unique value
    counts = np.bincount(dense_rank)
    csum = np.cumsum(counts)
    start = np.r_[0, csum[:-1]]
    ranks = (start + csum - 1) / 2.0 + 1.0
    return ranks[dense_rank][inv]

def log_batch_debug(pl_module, y_true, y_pred, mask=None, prefix="validation"):

    import torch
    yt = _to_numpy(y_true).ravel()
    yp = _to_numpy(y_pred).ravel()
    if mask is not None:
        m = _to_numpy(mask).astype(bool).ravel()
        valid = np.isfinite(yt) & np.isfinite(yp) & m
    else:
        valid = np.isfinite(yt) & np.isfinite(yp)

    n_total = yt.size
    n_valid = int(valid.sum())
    frac_valid = float(n_valid / max(1, n_total))

    yt_v = yt[valid]
    yp_v = yp[valid]

    yt_std = float(np.std(yt_v)) if n_valid > 0 else float("nan")
    yp_std = float(np.std(yp_v)) if n_valid > 0 else float("nan")

    pear = safe_corr(yt_v, yp_v, method="pearson") if n_valid > 2 else float("nan")
    spear = safe_corr(yt_v, yp_v, method="spearman") if n_valid > 2 else float("nan")

    # Use pl_module.log if available; else print.
    def _log(name, value):
        try:
            pl_module.log(name, value, prog_bar=False, on_epoch=True, on_step=False, sync_dist=False)
        except Exception:
            pass

    _log(f"{prefix}/debug_n_total", float(n_total))
    _log(f"{prefix}/debug_n_valid", float(n_valid))
    _log(f"{prefix}/debug_frac_valid", float(frac_valid))
    _log(f"{prefix}/debug_target_std", float(yt_std))
    _log(f"{prefix}/debug_pred_std", float(yp_std))
    _log(f"{prefix}/debug_pearson", float(pear if np.isfinite(pear) else 0.0))
    _log(f"{prefix}/debug_spearman", float(spear if np.isfinite(spear) else 0.0))

    # Also print once for visibility
    print(f"[{prefix} debug] n_total={n_total} n_valid={n_valid} (frac={frac_valid:.3f}) "
          f"target_std={yt_std:.4g} pred_std={yp_std:.4g} "
          f"pearson={pear} spearman={spear}")


def maybe_log_first_batch_debug(pl_module, batch, logits, targets, prefix="validation", batch_idx: int = 0):
    """Print one-time stats for the first batch of a split.

    Reports: valid_count, target_std, pred_std, and fraction of zero-embedding rows.

    - batch: collated dict from dataset (expects key 'token_ids': [B, L, D])
    - logits, targets: per-example tensors (typically already filtered to valid positions)
    - prefix: e.g., 'validation', 'fold_test', 'superfamily_test'
    - batch_idx: DataLoader batch index
    """
    import torch

    try:
        flag_name = f"_stb_first_batch_debug_done_{prefix}"
        if batch_idx != 0 or getattr(pl_module, flag_name, False):
            return

        feats = None
        try:
            if isinstance(batch, dict):
                feats = batch.get("token_ids", None)
        except Exception:
            feats = None

        zero_row_frac = "NA"
        if isinstance(feats, torch.Tensor) and feats.ndim == 3 and feats.numel() > 0:
            # rows (along D) that sum to zero indicate padding
            zero_rows = (feats.abs().sum(dim=-1) == 0)  # [B, L]
            zero_row_frac = float(zero_rows.float().mean().item())

        valid_count = -1
        pred_std = float("nan")
        target_std = float("nan")

        if isinstance(targets, torch.Tensor):
            valid_count = int(targets.numel())
            target_std = float(targets.detach().float().std().item()) if targets.numel() > 0 else float("nan")
        if isinstance(logits, torch.Tensor) and logits.numel() > 0:
            pred_std = float(logits.detach().float().std().item())

        print(
            f"[DEBUG][first_batch] split={prefix} valid_count={valid_count} "
            f"target_std={target_std:.6f} pred_std={pred_std:.6f} zero_row_frac={zero_row_frac}"
        )

    except Exception as e:
        try:
            print(f"[DEBUG][first_batch] failed to log: {e}")
        except Exception:
            pass
    finally:
        try:
            setattr(pl_module, flag_name, True)
        except Exception:
            pass


def log_every_batch_debug(pl_module, batch, logits, targets, prefix="validation", batch_idx: int = 0):
    """Log debug stats for every batch (no gating): valid_count, stds, zero-embed fraction.

    - batch: expects key 'token_ids' with shape [B, L, D]
    - logits/targets: tensors (any shape); std computed over all elements
    """
    import torch

    feats = None
    try:
        if isinstance(batch, dict):
            feats = batch.get("token_ids", None)
    except Exception:
        feats = None

    zero_row_frac = "NA"
    if isinstance(feats, torch.Tensor) and feats.ndim == 3 and feats.numel() > 0:
        zero_rows = (feats.abs().sum(dim=-1) == 0)  # [B, L]
        zero_row_frac = float(zero_rows.float().mean().item())

    valid_count = -1
    pred_std = float("nan")
    target_std = float("nan")

    if isinstance(targets, torch.Tensor):
        valid_count = int(targets.numel())
        if targets.numel() > 0:
            target_std = float(targets.detach().float().std().item())
    if isinstance(logits, torch.Tensor) and logits.numel() > 0:
        pred_std = float(logits.detach().float().std().item())

    try:
        print(
            f"[DEBUG][batch] split={prefix} idx={batch_idx} valid_count={valid_count} "
            f"target_std={target_std:.6f} pred_std={pred_std:.6f} zero_row_frac={zero_row_frac}"
        )
    except Exception:
        pass
