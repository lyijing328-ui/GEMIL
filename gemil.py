"""GEMIL training and checkpoint evaluation on pre-extracted medical-SSL bags."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
import yaml
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from architecture.moe_stkim import MoEACMIL
from data import build_datasets
from engine import evaluate, train_one_epoch
from utils.utils import FocalLoss, Struct, save_model, set_seed


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    conf = Struct(**values)
    required = {
        "pretrain": "medical_ssl", "D_feat": 384, "D_inner": 128,
        "B": 1, "arch": "moe", "n_token": 4,
        "use_stratified_stkim": True, "tam_mode": "mask",
        "tam_apply_in_eval": False, "n_shot": -1,
    }
    for name, value in required.items():
        if getattr(conf, name, None) != value:
            raise ValueError(f"{name} must be {value!r} in this release")
    classes = {"camelyon": 2, "bracs": 3, "nsclc": 2}
    if conf.dataset not in classes or conf.n_class != classes[conf.dataset]:
        raise ValueError("Unsupported dataset or class count")
    if not isinstance(conf.train_epoch, int) or not isinstance(conf.n_worker, int) or conf.train_epoch < 1 or conf.n_worker < 0:
        raise ValueError("train_epoch must be positive; n_worker must be nonnegative")
    for name in ("lr", "wd", "min_lr", "warmup_epoch"):
        value = getattr(conf, name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if conf.warmup_epoch >= conf.train_epoch:
        raise ValueError("warmup_epoch must be smaller than train_epoch")
    return conf


def make_criterion(conf, train_data, device):
    # Same training-set inverse-frequency weights and gamma as the original entry.
    if conf.use_focal_loss:
        labels = [train_data.data_dict[name]["label"] for name in train_data.data_names]
        counts = Counter(labels)
        alpha = [len(labels) / (conf.n_class * counts[i]) if i in counts else 1.0
                 for i in range(conf.n_class)]
        return FocalLoss(alpha=torch.tensor(alpha, dtype=torch.float, device=device), gamma=2.0)
    return nn.CrossEntropyLoss()


def load_checkpoint(path, model, conf):
    # Legacy project checkpoints contain utils.utils.Struct under the config key.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise ValueError("Expected a project checkpoint with a 'model' state_dict")
    saved = checkpoint.get("config", {})
    saved = vars(saved) if hasattr(saved, "__dict__") else saved
    for key in ("dataset", "pretrain", "D_feat", "D_inner", "n_class", "n_token",
                "use_focal_loss", "use_stratified_stkim", "stratified_group_ratios",
                "stratified_mask_rates"):
        if key in saved and saved[key] != getattr(conf, key):
            raise ValueError(f"Checkpoint/config mismatch for {key}: {saved[key]!r}")
    if saved.get("tam_mode", "mask") != conf.tam_mode:
        raise ValueError("Checkpoint TAM mode differs from config")
    if saved.get("tam_apply_in_eval", False) != conf.tam_apply_in_eval:
        raise ValueError("Checkpoint evaluation TAM setting differs from config")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Dataset YAML config")
    parser.add_argument("--features", required=True, help="Aggregate HDF5 feature file")
    parser.add_argument("--splits", required=True, help="Fixed train/val/test CSV")
    parser.add_argument("--output", help="New run directory (required for training)")
    parser.add_argument("--seed", type=int, default=None, help="Override YAML random seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a saved checkpoint")
    parser.add_argument("--checkpoint", help="Checkpoint to evaluate")
    args = parser.parse_args()
    if args.eval_only and not args.checkpoint:
        parser.error("--eval-only requires --checkpoint")
    if not args.eval_only and (not args.output or args.checkpoint):
        parser.error("training requires --output; --checkpoint is used with --eval-only")
    return args


def main():
    args = arguments()
    conf = load_config(args.config)
    if args.seed is not None:
        conf.seed = args.seed
    set_seed(conf.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    train_data, val_data, test_data = build_datasets(args.features, args.splits, conf)
    loaders = [DataLoader(dataset, batch_size=conf.B, shuffle=(i == 0),
                          num_workers=conf.n_worker, pin_memory=conf.pin_memory,
                          drop_last=(i == 0))
               for i, dataset in enumerate((train_data, val_data, test_data))]
    model = MoEACMIL(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch,
                     mask_drop=conf.mask_drop, n_expand=0).to(device)
    criterion = make_criterion(conf, train_data, device)
    if args.eval_only:
        checkpoint = load_checkpoint(args.checkpoint, model, conf)
        result = {"epoch": checkpoint.get("epoch", -1)}
        for name, loader in zip(("val", "test"), loaders[1:]):
            result[name] = dict(zip(("auc", "acc_percent", "f1", "loss"),
                                    evaluate(model, criterion, loader, device, conf, name)))
        print(json.dumps(result, indent=2))
        return

    from torch.utils.tensorboard import SummaryWriter
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    conf.features = str(Path(args.features).resolve())
    conf.splits = str(Path(args.splits).resolve())
    (output / "config.json").write_text(json.dumps(vars(conf), indent=2), encoding="utf-8")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=0.001, weight_decay=conf.wd)
    best_state = {"epoch": -1, "val_acc": 0, "val_auc": 0, "val_f1": 0,
                  "test_acc": 0, "test_auc": 0, "test_f1": 0}
    with SummaryWriter(log_dir=str(output / "tensorboard")) as writer:
        for epoch in range(conf.train_epoch):
            train_one_epoch(model, criterion, loaders[0], optimizer, device, epoch, conf, writer)
            val_auc, val_acc, val_f1, val_loss = evaluate(model, criterion, loaders[1], device, conf, "Val")
            test_auc, test_acc, test_f1, test_loss = evaluate(model, criterion, loaders[2], device, conf, "Test")
            for split, metrics in (("val", (val_acc, val_auc, val_f1, val_loss)),
                                   ("test", (test_acc, test_auc, test_f1, test_loss))):
                for metric, value in zip(("acc1", "auc", "f1", "loss"), metrics):
                    writer.add_scalar(f"perf/{split}_{metric}", value, epoch)
            # Preserve the actual historical selection rule.
            if test_f1 + test_auc > best_state["test_f1"] + best_state["test_auc"]:
                best_state.update(epoch=epoch, val_auc=val_auc, val_acc=val_acc, val_f1=val_f1,
                                  test_auc=test_auc, test_acc=test_acc, test_f1=test_f1)
                save_model(conf=conf, model=model, optimizer=optimizer, epoch=epoch,
                           save_path=str(output / "checkpoint-best.pth"))
        save_model(conf=conf, model=model, optimizer=optimizer, epoch=epoch,
                   save_path=str(output / "checkpoint-last.pth"))
    (output / "best_metrics.json").write_text(json.dumps(best_state, indent=2), encoding="utf-8")
    logger.info("Results on best epoch: {}", best_state)


if __name__ == "__main__":
    main()
