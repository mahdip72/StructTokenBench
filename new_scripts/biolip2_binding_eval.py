#!/usr/bin/env python3
import argparse
import copy
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.biolip2 import BioLIP2FunctionDataset
from dataset.tokenizer_biolip2 import WrappedMyRepBioLIP2Tokenizer


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_auroc(scores: np.ndarray, targets: np.ndarray) -> float:
    scores = np.asarray(scores).reshape(-1)
    targets = np.asarray(targets).astype(np.int64).reshape(-1)
    if scores.size == 0:
        return float("nan")
    pos = targets.sum()
    neg = targets.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    y = targets[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / pos
    fpr = fps / neg
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return float(np.trapz(tpr, fpr))


def build_dataset(split: str, args, tokenizer, logger):
    logger.info(f"[data] init dataset split={split}")
    ds = BioLIP2FunctionDataset(
        data_path=args.data_root,
        split=split,
        target_field="binding_label",
        pdb_data_dir=args.pdb_dir,
        tokenizer=tokenizer,
        logger=logger,
        cache=True,
    )
    logger.info(f"[data] dataset split={split} size={len(ds)}")
    return ds


def make_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def _iter_with_progress(loader, enabled, desc):
    if not enabled:
        return loader
    try:
        from tqdm import tqdm
        return tqdm(loader, desc=desc, leave=False)
    except Exception:
        return loader


def train_one_epoch(model, loader, optimizer, device, show_progress=False):
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in _iter_with_progress(loader, show_progress, desc="train"):
        if batch is None:
            continue
        feats = batch["token_ids"].to(device)
        targets = batch["labels"].to(device)
        mask = targets != -100
        if mask.sum() == 0:
            continue
        logits = model(feats).squeeze(-1)
        logits = logits[mask]
        targets = targets[mask]
        pos = targets.sum()
        neg = targets.numel() - pos
        if pos > 0 and neg > 0:
            pos_weight = targets.numel() / pos * 0.5
            neg_weight = targets.numel() / neg * 0.5
            weights = torch.full_like(targets, neg_weight)
            weights[targets >= 0.5] = pos_weight
            loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(1, steps)


@torch.no_grad()
def eval_auroc(model, loader, device, show_progress=False, desc="eval"):
    model.eval()
    all_scores = []
    all_targets = []
    for batch in _iter_with_progress(loader, show_progress, desc=desc):
        if batch is None:
            continue
        feats = batch["token_ids"].to(device)
        targets = batch["labels"].to(device)
        mask = targets != -100
        if mask.sum() == 0:
            continue
        logits = model(feats).squeeze(-1)
        scores = torch.sigmoid(logits[mask]).detach().cpu().numpy()
        labs = targets[mask].detach().cpu().numpy()
        all_scores.append(scores)
        all_targets.append(labs)
    if not all_scores:
        return float("nan")
    scores = np.concatenate(all_scores, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return binary_auroc(scores, targets)


def parse_args():
    p = argparse.ArgumentParser(description="BioLIP2 binding site AUROC eval (minimal)")
    p.add_argument(
        "--h5",
        required=True,
        help="Path to BioLIP2 embedding H5 (e.g., vq_embed_biolip2_binding_lite_model.h5)",
    )
    p.add_argument(
        "--embeddings-dataset",
        default="/",
        help="H5 dataset/group path (default: '/')",
    )
    p.add_argument(
        "--data-root",
        default="/mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/functional/local",
        help="Root containing biolip2/processed_structured_* files",
    )
    p.add_argument(
        "--pdb-dir",
        default="/mnt/hdd8/farzaneh/projects/PST/mmcif_files/pdb_data",
        help="Directory that contains mmcif_files/ (or mmCIF files directly)",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokenizer-device", default="cpu")
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bars")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("biolip2_binding_eval")

    seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    tokenizer = WrappedMyRepBioLIP2Tokenizer(
        h5_path=args.h5,
        embeddings_dataset=args.embeddings_dataset,
        device=str(args.tokenizer_device),
    )

    train_ds = build_dataset("train", args, tokenizer, logger)
    val_ds = build_dataset("validation", args, tokenizer, logger)
    fold_ds = build_dataset("fold_test", args, tokenizer, logger)
    sfam_ds = build_dataset("superfamily_test", args, tokenizer, logger)

    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, shuffle=False)
    fold_loader = make_loader(fold_ds, args.batch_size, args.num_workers, shuffle=False)
    sfam_loader = make_loader(sfam_ds, args.batch_size, args.num_workers, shuffle=False)

    sample_dim = None
    for batch in train_loader:
        if batch is None:
            continue
        sample_dim = int(batch["token_ids"].shape[-1])
        break
    if sample_dim is None:
        raise RuntimeError("No valid samples in training set.")
    embed_dim = sample_dim
    model = nn.Linear(embed_dim, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger.info("[train] start")
    best_state = None
    best_val = float("-inf")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, show_progress=args.progress)
        val_auroc = eval_auroc(model, val_loader, device, show_progress=args.progress, desc="val")
        logger.info(f"[train] epoch={epoch+1} loss={loss:.4f} val_auroc={val_auroc*100:.2f}")
        if val_auroc > best_val:
            best_val = val_auroc
            best_state = copy.deepcopy(model.state_dict())
    logger.info("[train] finished")

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"[train] loaded best val checkpoint (val_auroc={best_val*100:.2f})")

    fold_auroc = eval_auroc(model, fold_loader, device, show_progress=args.progress, desc="fold_test")
    sfam_auroc = eval_auroc(model, sfam_loader, device, show_progress=args.progress, desc="superfamily_test")
    logger.info(f"[final] fold_test AUROC% = {fold_auroc*100:.2f}")
    logger.info(f"[final] superfamily_test AUROC% = {sfam_auroc*100:.2f}")


if __name__ == "__main__":
    main()
