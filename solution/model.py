"""Modelo de posesión: pseudo-etiquetas -> clasificador de forma -> evidencias -> Viterbi.

Ninguna etiqueta humana del partido se usa en ningún paso. Las pseudo-etiquetas salen de reglas
geométricas sobre el tracking automático (control claro del balón, balón fuera del campo) y de
las marcas automáticas de balón parado del Smart Tagging.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from .features import shape_features, _roll
from .io_data import FPS

H, A, D = 0, 1, 2
STATES = np.array(["H", "A", "D"])


# ------------------------------------------------------------------------------ pseudo-etiquetas
def _runs(mask):
    """(inicio, fin_exclusivo) de cada tramo True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def control_codes(df, near=1.2, far=2.0):
    """Por frame: H/A si un equipo controla claramente el balón (visible), -1 si no."""
    hd, ad = df["h_dball"].to_numpy(), df["a_dball"].to_numpy()
    present = df["ball_present"].to_numpy() > 0
    code = np.full(len(df), -1, np.int8)
    code[present & (hd < near) & ~(ad < far)] = H
    code[present & (ad < near) & ~(hd < far)] = A
    return code


def setpiece_windows(tags, n, before_s=3.0, after_s=0.5):
    """Máscara de frames justo antes del inicio de cada SET PIECES (balón parado esperando)."""
    m = np.zeros(n, bool)
    for t in tags:
        if t["code"] == "SET PIECES":
            a = max(0, t["start"] - int(before_s * FPS))
            b = min(n, t["start"] + int(after_s * FPS))
            m[a:b] = True
    return m


def pseudo_labels(df, tags, graded, min_run=10):
    n = len(df)
    y = np.full(n, -1, np.int8)
    code = control_codes(df)
    alive = _roll(df["speed_all"].to_numpy(), 50) > 1.3
    for team in (H, A):
        s, e = _runs((code == team) & alive)
        for a, b in zip(s, e):
            if b - a >= min_run:
                y[a:b] = team
    oob = np.nan_to_num(df["ball_oob"].to_numpy()) > 0
    sp = setpiece_windows(tags, n)
    y[(oob | sp)] = D
    y[~graded] = -1
    return y


# ------------------------------------------------------------------------------ clasificador
def train_shape_classifier(X, y, half, seed=0):
    """LightGBM 3 clases, validación cruzada por mitades (entrena en una, predice la otra).

    Así cada frame se predice con un modelo que no vio su mitad, lo que evita que el
    clasificador memorice sus propias pseudo-etiquetas.
    """
    proba = np.full((len(X), 3), np.nan, np.float32)
    params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=seed, num_threads=4,
                  deterministic=True, force_col_wise=True)
    for h in (1, 2):
        tr = (half != h) & (y >= 0)
        # submuestreo temporal: frames vecinos son casi idénticos
        idx = np.flatnonzero(tr)[::5]
        w = _class_weights(y[idx])
        ds = lgb.Dataset(X.iloc[idx], y[idx], weight=w)
        model = lgb.train(params, ds, num_boost_round=300)
        te = half == h
        proba[te] = model.predict(X[te])
    return proba


def _class_weights(y):
    counts = np.bincount(y, minlength=3).astype(float)
    w = counts.sum() / (3 * np.maximum(counts, 1))
    return w[y]


# ------------------------------------------------------------------------------ evidencias
def emissions(df, proba, tags, temp=1.0, ball_bonus=2.0, oob_bonus=3.0, sp_bonus=1.0):
    """Log-verosimilitudes por frame (n, 3) combinando clasificador y balón."""
    lp = np.log(np.clip(proba, 1e-4, 1)) / temp
    code = control_codes(df, near=1.5, far=2.5)
    lp[code == H, H] += ball_bonus
    lp[code == A, A] += ball_bonus
    oob = np.nan_to_num(df["ball_oob"].to_numpy()) > 0
    lp[oob, D] += oob_bonus
    lp[setpiece_windows(tags, len(df), 2.0, 0.0), D] += sp_bonus
    return lp


def viterbi(lp, switch_cost, dead_cost):
    """Viterbi de 3 estados. switch_cost: H<->A; dead_cost: entrar o salir de D."""
    n = len(lp)
    trans = np.array([[0, -switch_cost, -dead_cost],
                      [-switch_cost, 0, -dead_cost],
                      [-dead_cost, -dead_cost, 0]], np.float64)
    score = lp[0].astype(np.float64).copy()
    back = np.zeros((n, 3), np.int8)
    for t in range(1, n):
        cand = score[:, None] + trans          # cand[i, j]: de i a j
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + lp[t]
    path = np.zeros(n, np.int8)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(lp, blocks, switch_cost, dead_cost):
    """Decodifica cada mitad por separado; fuera de las mitades, X."""
    labels = np.full(len(lp), "X", dtype="<U1")
    for _, a, b in blocks:
        labels[a:b + 1] = STATES[viterbi(lp[a:b + 1], switch_cost, dead_cost)]
    return labels


def build_features(df):
    return shape_features(df)
