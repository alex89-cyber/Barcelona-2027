"""Carga del tracking (ATD), Smart Tagging y homografía en arrays numpy, con caché.

La caché se guarda en data/cache/ (gitignored). Nunca se sube a git: es derivada de los datos.
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache")
PITCH = GAMES[GAME]["pitch"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]


def track_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], np.int8)


def _parse_raw(raw_path, n_slots, n_frames):
    players = np.full((n_frames, n_slots, 3), np.nan, np.float32)
    ball = np.full((n_frames, 3), np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            if not 0 <= f < n_frames:
                continue
            vals = parts[1].replace(";", ",").split(",")
            players[f] = np.array(vals, dtype=np.float32).reshape(n_slots, 3)
            ball[f] = np.array(parts[2].split(","), dtype=np.float32)
    return players, ball


def load_tracking(source="tactical", game=GAME):
    """Devuelve dict con players (F,S,3) en metros (x,y,speed), ball (F,3) y team (S,)."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "{}_{}_atd.npz".format(game, source))
    paths = asset_paths(source=source, game=game)
    team = track_teams(paths["atd_meta"])
    if not os.path.exists(cache):
        n = N_FRAMES if source == "tactical" else sum(1 for _ in open(paths["atd_raw"]))
        players, ball = _parse_raw(paths["atd_raw"], len(team), n)
        np.savez_compressed(cache, players=players, ball=ball)
    d = np.load(cache)
    players, ball = d["players"].copy(), d["ball"].copy()
    for arr in (players, ball):
        arr[..., 0] *= PITCH[0]
        arr[..., 1] *= PITCH[1]
    return {"players": players, "ball": ball, "team": team}


def load_tags(game=GAME):
    """Smart Tagging como lista de dicts: start/end en frames de vídeo, code y tags (minúsculas)."""
    root = ET.parse(asset_paths(game=game)["smart_tagging"]).getroot()
    out = []
    for inst in root.iter("instance"):
        tags = {}
        for lab in inst.iter("label"):
            g = (lab.findtext("group") or "").strip().lower()
            tags[g] = (lab.findtext("text") or "").strip().lower()
        out.append({
            "start": int(round(float(inst.findtext("start")) * FPS)),
            "end": int(round(float(inst.findtext("end")) * FPS)),
            "code": (inst.findtext("code") or "").strip().upper(),
            "tags": tags,
        })
    return out


def graded_mask():
    m = np.zeros(N_FRAMES, bool)
    for _, a, b in half_video_blocks(GAME):
        m[a:b + 1] = True
    return m
