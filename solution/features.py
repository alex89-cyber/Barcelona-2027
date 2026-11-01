"""Features por frame a partir del tracking táctico.

Todo se calcula de forma algorítmica a partir del ATD; no se usa ninguna etiqueta humana.

Convenciones
------------
- Coordenadas en metros, (0,0) en una esquina, campo de 105.3 x 68.
- Para cada equipo se define x_att: la coordenada x vista en su dirección de ataque (0 = su
  portería, 105.3 = la portería rival). La dirección se infiere por mitad: el equipo con la x
  media más baja defiende la portería izquierda.
"""
import numpy as np
import pandas as pd

from .io_data import FPS, N_FRAMES, PITCH, half_video_blocks, GAME

L, W = PITCH
HOME, AWAY = 0, 1


def _nanfill_interp(a, max_gap):
    """Interpola linealmente huecos de NaN de hasta max_gap frames."""
    s = pd.Series(a)
    return s.interpolate(limit=max_gap, limit_area="inside").to_numpy()


def _roll(a, win, fn="mean"):
    s = pd.Series(a)
    r = s.rolling(win, center=True, min_periods=1)
    return getattr(r, fn)().to_numpy()


def half_of_frame():
    h = np.zeros(N_FRAMES, np.int8)
    for half, a, b in half_video_blocks(GAME):
        h[a:b + 1] = half
    # Fuera de las mitades se usa la mitad más cercana (no se puntúa, solo evita NaN).
    blocks = half_video_blocks(GAME)
    h[:blocks[0][1]] = 1
    h[blocks[0][2] + 1:] = 2
    h[blocks[0][2] + 1:blocks[1][1]] = 1
    return h


def attack_sign(players, team):
    """Para cada mitad y equipo: +1 si ataca hacia x creciente, -1 si hacia x decreciente."""
    signs = {}
    for half, a, b in half_video_blocks(GAME):
        mx = [np.nanmean(players[a:b + 1, team == t, 0]) for t in (HOME, AWAY)]
        home_sign = 1 if mx[HOME] < mx[AWAY] else -1
        signs[half] = {HOME: home_sign, AWAY: -home_sign}
    return signs


def compute(track):
    P, B, team = track["players"], track["ball"], track["team"]
    n = len(B)
    half = half_of_frame()
    signs = attack_sign(P, team)
    sgn = np.zeros((n, 2), np.float32)
    for hf in (1, 2):
        m = half == hf
        sgn[m, HOME] = signs[hf][HOME]
        sgn[m, AWAY] = signs[hf][AWAY]

    f = {}
    # ---------------------------------------------------------------- velocidades de jugadores
    xy = P[:, :, :2]
    v = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[:1]), axis=2) * FPS
    v[v > 12] = np.nan  # saltos de identidad del tracker
    vel = np.diff(xy, axis=0, prepend=xy[:1]) * FPS
    vel[np.linalg.norm(vel, axis=2) > 12] = np.nan

    with np.errstate(all="ignore"):
        all_speed = np.nanmean(v, axis=1)
        f["speed_all"] = all_speed
        f["slow_frac"] = np.nanmean(np.where(np.isnan(v), np.nan, v < 1.0), axis=1)
        f["n_tracked"] = np.sum(~np.isnan(P[:, :, 0]), axis=1).astype(np.float32)

        for t, name in ((HOME, "h"), (AWAY, "a")):
            m = team == t
            x = P[:, m, 0]
            y = P[:, m, 1]
            s = sgn[:, t][:, None]
            x_att = np.where(s > 0, x, L - x)
            f[f"{name}_n"] = np.sum(~np.isnan(x), axis=1).astype(np.float32)
            f[f"{name}_cx"] = np.nanmean(x_att, axis=1)
            f[f"{name}_cy"] = np.nanmean(y, axis=1)
            f[f"{name}_sx"] = np.nanstd(x_att, axis=1)
            f[f"{name}_sy"] = np.nanstd(y, axis=1)
            f[f"{name}_maxx"] = np.nanmax(x_att, axis=1)
            f[f"{name}_minx"] = np.nanmin(x_att, axis=1)
            f[f"{name}_speed"] = np.nanmean(v[:, m], axis=1)
            vx = vel[:, m, 0] * s
            f[f"{name}_vx"] = np.nanmean(vx, axis=1)        # velocidad hacia portería rival
            f[f"{name}_vy_abs"] = np.nanmean(np.abs(vel[:, m, 1]), axis=1)
            f[f"{name}_in_opp_half"] = np.nansum(x_att > L / 2, axis=1).astype(np.float32)

    # ---------------------------------------------------------------- balón
    bx_raw, by_raw = B[:, 0], B[:, 1]
    f["ball_present"] = (~np.isnan(bx_raw)).astype(np.float32)
    bx = _nanfill_interp(bx_raw, 12)
    by = _nanfill_interp(by_raw, 12)
    bxs = _roll(bx, 5)
    bys = _roll(by, 5)
    bv = np.hypot(np.gradient(bxs), np.gradient(bys)) * FPS
    f["ball_x"], f["ball_y"], f["ball_speed"] = bxs, bys, bv
    f["ball_oob"] = ((bx < -0.5) | (bx > L + 0.5) | (by < -0.5) | (by > W + 0.5)).astype(np.float32)
    f["ball_oob"][np.isnan(bx)] = np.nan

    dist = np.linalg.norm(xy - np.stack([bx, by], 1)[:, None, :], axis=2)
    for t, name in ((HOME, "h"), (AWAY, "a")):
        d_t = dist[:, team == t]
        with np.errstate(all="ignore"):
            f[f"{name}_dball"] = np.nanmin(np.where(np.isnan(d_t), np.inf, d_t), axis=1)
        f[f"{name}_dball"][np.isinf(f[f"{name}_dball"])] = np.nan
        f[f"{name}_dball"][np.isnan(bx)] = np.nan
    # Ball x en coordenadas de ataque del local (para el clasificador de balón parado)
    f["ball_x_home_att"] = np.where(sgn[:, HOME] > 0, bxs, L - bxs)

    df = pd.DataFrame(f)
    df["half"] = half
    df["sgn_home"] = sgn[:, HOME]
    return df


def shape_features(df):
    """Features de forma de equipos (sin balón) + contexto temporal, para el clasificador H/A."""
    base = ["h_cx", "a_cx", "h_sx", "a_sx", "h_sy", "a_sy", "h_maxx", "a_maxx", "h_minx",
            "a_minx", "h_speed", "a_speed", "h_vx", "a_vx", "h_vy_abs", "a_vy_abs",
            "h_in_opp_half", "a_in_opp_half", "h_n", "a_n", "speed_all"]
    out = {}
    for c in base:
        col = df[c].to_numpy()
        for w in (1, 25, 75, 200):
            out[f"{c}_r{w}"] = col if w == 1 else _roll(col, w)
    o = pd.DataFrame(out)
    for w in (1, 25, 75, 200):
        o[f"d_cx_r{w}"] = o[f"h_cx_r{w}"] - o[f"a_cx_r{w}"]
        o[f"d_sx_r{w}"] = o[f"h_sx_r{w}"] - o[f"a_sx_r{w}"]
        o[f"d_sy_r{w}"] = o[f"h_sy_r{w}"] - o[f"a_sy_r{w}"]
        o[f"d_vx_r{w}"] = o[f"h_vx_r{w}"] - o[f"a_vx_r{w}"]
        o[f"d_opp_r{w}"] = o[f"h_in_opp_half_r{w}"] - o[f"a_in_opp_half_r{w}"]
    # tendencias: cambio de centroide en las próximas/pasadas ventanas
    for c in ("h_cx", "a_cx"):
        col = _roll(df[c].to_numpy(), 25)
        for lag in (25, 75):
            o[f"{c}_fwd{lag}"] = np.roll(col, -lag) - col
            o[f"{c}_bwd{lag}"] = col - np.roll(col, lag)
    return o.astype(np.float32)
