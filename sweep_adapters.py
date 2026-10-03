"""Test combinations of adapter positions, ranks and learning rates on Apertus.

Each run attaches one combination, trains it on --train, measures the loss on --val,
then removes the adapters again. Results are appended to a CSV, so an interrupted
sweep can be restarted and will skip the runs already done.

  python sweep_adapters.py --train train.jsonl --val val.jsonl                 # all 63 combinations
  python sweep_adapters.py --train train.jsonl --val val.jsonl --max-size 2    # singles and pairs (21)
  python sweep_adapters.py --train train.jsonl --val val.jsonl \
      --positions mlp attention embed --ranks 16 64 --lrs 1e-4 3e-4            # 7 x 2 x 2 = 28 runs
"""
import argparse
import csv
import itertools
import os
import time

from apertus_adapters import (POSITIONS, add_adapters, count_params, evaluate, load_model,
                              read_jsonl, remove_adapters, save_adapters, train_adapters)

FIELDS = ["positions", "rank", "lr", "val_loss", "trainable_params", "minutes"]


def combinations(positions, min_size=1, max_size=None):
    max_size = max_size or len(positions)
    return [c for n in range(min_size, max_size + 1) for c in itertools.combinations(positions, n)]


def run_sweep(model, tok, train_rows, val_rows, combos, ranks, lrs, out="sweep_results.csv",
              epochs=1, accum=8, max_len=8192, save_dir=None):
    done = set()
    if os.path.exists(out):
        done = {(r["positions"], r["rank"], r["lr"]) for r in csv.DictReader(open(out))}
    else:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS); w.writeheader()
            w.writerow(dict(positions="none (base model)", rank=0, lr=0,
                            val_loss=round(evaluate(model, tok, val_rows, max_len), 4),
                            trainable_params=0, minutes=0))
    runs = [(c, r, lr) for c in combos for r in ranks for lr in lrs]
    for i, (combo, rank, lr) in enumerate(runs, 1):
        key = ("+".join(combo), str(rank), str(lr))
        if key in done:
            continue
        t0 = time.time()
        add_adapters(model, r=rank, positions=combo)
        train_adapters(model, tok, train_rows, lr, epochs, accum, max_len, log=False)
        row = dict(positions=key[0], rank=rank, lr=lr,
                   val_loss=round(evaluate(model, tok, val_rows, max_len), 4),
                   trainable_params=count_params(model)[1], minutes=round((time.time() - t0) / 60, 2))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_adapters(model, os.path.join(save_dir, f"{key[0]}_r{rank}_lr{lr}.pt"), r=rank, positions=combo)
        remove_adapters(model)
        with open(out, "a", newline="") as f:
            csv.DictWriter(f, FIELDS).writerow(row)
        print(f"[{i}/{len(runs)}] {row}")
    rows = sorted(csv.DictReader(open(out)), key=lambda r: float(r["val_loss"]))
    print("\nBest runs (lowest validation loss):")
    for r in rows[:10]:
        print(f"  {float(r['val_loss']):.4f}  {r['positions']:<45} rank {r['rank']:<4} lr {r['lr']}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="swiss-ai/Apertus-8B-Instruct-2509")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--positions", nargs="+", default=list(POSITIONS), choices=POSITIONS)
    ap.add_argument("--min-size", type=int, default=1)
    ap.add_argument("--max-size", type=int, default=None)
    ap.add_argument("--ranks", nargs="+", type=int, default=[64])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-4])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--out", default="sweep_results.csv")
    ap.add_argument("--save-dir", default=None, help="also save each run's adapter weights here")
    ap.add_argument("--four-bit", action="store_true")
    a = ap.parse_args()
    combos = combinations(a.positions, a.min_size, a.max_size)
    print(f"{len(combos)} combinations x {len(a.ranks)} ranks x {len(a.lrs)} learning rates "
          f"= {len(combos) * len(a.ranks) * len(a.lrs)} runs")
    tok, model = load_model(a.model, a.four_bit)
    run_sweep(model, tok, read_jsonl(a.train), read_jsonl(a.val), combos, a.ranks, a.lrs,
              a.out, a.epochs, a.accum, a.max_len, a.save_dir)
