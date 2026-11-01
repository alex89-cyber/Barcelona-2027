"""Genera la entrega completa a partir de los datos en ./data.

    python -m solution.run                     # escribe submissions/<equipo>.csv
    python -m solution.run --team mi_equipo --out evaluation/results

Determinista: misma entrada -> mismo CSV (semillas fijas, un hilo de decisión por paso).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from .io_data import ROOT, CACHE, GAME, load_tracking, load_tags, graded_mask, half_video_blocks
from .features import compute
from . import model as M

# Hiperparámetros fijados a priori (ver docs/): no se ajustan contra el GT público.
PARAMS = dict(temp=1.0, ball_bonus=2.0, oob_bonus=3.0, sp_bonus=1.0,
              switch_cost=12.0, dead_cost=12.0)


def run(params=PARAMS, verbose=True):
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    feat_cache = os.path.join(CACHE, "frame_features.parquet")
    if os.path.exists(feat_cache):
        df = pd.read_parquet(feat_cache)
    else:
        df = compute(load_tracking())
        df.to_parquet(feat_cache)
    tags = load_tags()
    graded = graded_mask()
    log("features por frame: {:.1f}s".format(time.time() - t0))

    y = M.pseudo_labels(df, tags, graded)
    log("pseudo-etiquetas: H {} A {} D {} (de {} frames puntuables)".format(
        *(int((y == k).sum()) for k in range(3)), int(graded.sum())))

    X = M.build_features(df)
    proba_cache = os.path.join(CACHE, "shape_proba.npy")
    proba = M.train_shape_classifier(X, y, df["half"].to_numpy())
    np.save(proba_cache, proba)
    log("clasificador: {:.1f}s".format(time.time() - t0))

    lp = M.emissions(df, proba, tags, params["temp"], params["ball_bonus"],
                     params["oob_bonus"], params["sp_bonus"])
    labels = M.decode(lp, half_video_blocks(GAME), params["switch_cost"], params["dead_cost"])
    log("decodificado: {:.1f}s".format(time.time() - t0))
    return labels, dict(df=df, y=y, proba=proba, lp=lp)


def write(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    return path


def summary(labels, graded):
    lab = labels[graded]
    counts = {k: int((lab == k).sum()) for k in "HAD"}
    spells = 0
    for _, a, b in half_video_blocks(GAME):
        seg = labels[a:b + 1]
        change = np.r_[True, seg[1:] != seg[:-1]]
        spells += int(np.sum(change & (seg != "D")))
    return counts, spells


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="barcelona2027")
    ap.add_argument("--out", default=os.path.join(ROOT, "submissions"))
    args = ap.parse_args()
    labels, _ = run()
    counts, spells = summary(labels, graded_mask())
    print("frames:", counts, "posesiones:", spells)
    print("->", write(os.path.join(args.out, args.team + ".csv"), labels))
