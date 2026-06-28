import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ba_mpd.datasets import BeansCBIParquetDataset, collate_waveforms
from ba_mpd.losses import ba_mpd_loss, mpd_loss
from ba_mpd.metrics import classification_metrics
from ba_mpd.models import PannsCnn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(csv_path, args, shuffle):
    dataset = BeansCBIParquetDataset(
        csv_path,
        data_dir=args.data_dir,
        label_mapping_csv=args.label_mapping_csv,
        sample_rate=args.sample_rate,
        duration=args.duration,
        row_group_cache_size=args.row_group_cache_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0 and args.persistent_workers,
        collate_fn=collate_waveforms,
    )
    return loader, dataset


def load_teacher_logits(args, train_dataset):
    logits = torch.from_numpy(np.load(args.teacher_logits_npy)).float()
    with open(args.teacher_logits_index) as handle:
        index = json.load(handle)
    if isinstance(index, dict) and "sample_ids" in index:
        sample_ids = index["sample_ids"]
    elif isinstance(index, list):
        sample_ids = index
    else:
        raise ValueError(f"Unsupported teacher-logits index format: {args.teacher_logits_index}")
    position = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    order = [position[sample_id] for sample_id in train_dataset.frame["sample_id"].tolist()]
    return logits.index_select(0, torch.as_tensor(order, dtype=torch.long))


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_true = []
    all_pred = []
    for waveforms, labels, _ in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(waveforms)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += float(loss.detach().cpu())
        total_count += int(labels.numel())
        all_true.append(labels.detach().cpu().numpy())
        all_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    metrics = classification_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def distillation_loss(student_logits, teacher_logits, labels, args):
    if args.objective == "mpd":
        return mpd_loss(
            student_logits,
            teacher_logits,
            top_k=args.top_k,
            temperature=args.temperature,
            alpha=args.alpha,
        )
    if args.objective == "ba-mpd":
        return ba_mpd_loss(
            student_logits,
            teacher_logits,
            labels,
            top_k=args.top_k,
            temperature=args.temperature,
            alpha=args.alpha,
        )
    raise ValueError(f"Unknown distillation objective: {args.objective}")


def train(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader, train_dataset = build_loader(args.train_csv, args, shuffle=True)
    valid_loader, _ = build_loader(args.valid_csv, args, shuffle=False)
    test_loader, _ = build_loader(args.test_csv, args, shuffle=False)
    teacher_logits = load_teacher_logits(args, train_dataset)
    sample_to_pos = {sid: i for i, sid in enumerate(train_dataset.frame["sample_id"].tolist())}

    model = PannsCnn(n_classes=args.n_classes, n_mels=args.n_mels, sample_rate=args.sample_rate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = min(args.warmup_steps, total_steps // 2)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_macro = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for waveforms, labels, sample_ids in train_loader:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(waveforms)
                ce_loss = F.cross_entropy(logits, labels)
                positions = torch.as_tensor([sample_to_pos[sid] for sid in sample_ids], dtype=torch.long)
                batch_teacher = teacher_logits.index_select(0, positions).to(device, non_blocking=True)
                kd_loss = distillation_loss(logits, batch_teacher, labels, args)
                loss = ce_loss + float(args.kd_weight) * kd_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += float(loss.detach().cpu()) * int(labels.numel())
            train_count += int(labels.numel())

        valid = evaluate(model, valid_loader, device, args.n_classes)
        record = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": train_loss / max(train_count, 1),
            "valid": valid,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if valid["macro_accuracy"] > best_macro:
            best_macro = valid["macro_accuracy"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "valid": valid, "args": vars(args)},
                output_dir / "best.pt",
            )

    best = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"], strict=True)
    test = evaluate(model, test_loader, device, args.n_classes)
    summary = {
        "objective": args.objective,
        "budget": args.budget,
        "best_epoch": int(best["epoch"]),
        "best_valid": best["valid"],
        "test": test,
        "primary_metrics": ["macro_accuracy", "macro_f1", "micro_accuracy"],
        "model_selection": "official validation macro_accuracy",
        "args": vars(args),
        "history": history,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"summary": str(output_dir / "summary.json"), "test": test}, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description="Train BEANS-CBI students with MPD or BA-MPD.")
    parser.add_argument("--objective", choices=["mpd", "ba-mpd"], default="ba-mpd")
    parser.add_argument("--data-dir", required=True, help="Directory containing BEANS-CBI parquet shards.")
    parser.add_argument("--train-csv", default="splits/beans_cbi/train_10.csv")
    parser.add_argument("--valid-csv", default="splits/beans_cbi/valid.csv")
    parser.add_argument("--test-csv", default="splits/beans_cbi/test.csv")
    parser.add_argument("--label-mapping-csv", default="splits/beans_cbi/label_mapping.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-logits-npy", default=None)
    parser.add_argument("--teacher-logits-index", default=None)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-classes", type=int, default=264)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--row-group-cache-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kd-weight", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.teacher_logits_npy or not args.teacher_logits_index:
        raise ValueError("MPD/BA-MPD require --teacher-logits-npy and --teacher-logits-index")
    return args


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
