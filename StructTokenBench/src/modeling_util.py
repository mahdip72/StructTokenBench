import torch
import deepspeed
import contextlib

from util import get_dtype

def model_init_fn(trainer, model_cfg, **model_kwargs):
    """deepspeed-compatible model initialization
    Do not depend on this for loading model state_dict
    Use `ckpt_path` for lightning trainer instead
    """

    init_dtype = get_dtype(trainer.precision)

    # Safely detect DeepSpeed config; default to a no-op when not using DeepSpeed
    strategy = getattr(trainer, "strategy", None)
    ds_cfg = getattr(strategy, "config", None)
    remote_device = getattr(strategy, "remote_device", None)

    is_zero3 = False
    if isinstance(ds_cfg, dict):
        try:
            is_zero3 = (ds_cfg.get("zero_optimization", {}).get("stage", 0) == 3)
        except Exception:
            is_zero3 = False

    if isinstance(ds_cfg, dict):
        context = deepspeed.zero.Init(
            remote_device=remote_device,
            pin_memory=True,
            config_dict_or_path=ds_cfg,
            dtype=init_dtype,
            enabled=is_zero3,
        )
    else:
        # Not DeepSpeed: no-op context
        context = contextlib.nullcontext()
    
    from model_module import SequenceClassificationModel, ZeroshotProximityModel, ZeroShotCodebookUtilityModel
    from vqvae_model import VQVAEModel

    with context:
        model = eval(model_cfg.class_name)(
            model_cfg,
            **model_kwargs
        )

    return model

