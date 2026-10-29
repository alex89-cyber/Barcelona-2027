# Barcelona 2027 · More than a Hack 2027

Nuestro trabajo para **More than a Hack 2027** (Mobile World Capital Barcelona, Barça Innovation Hub y Metrica Sports).

**Fase 1 (posesión del balón):** etiquetar cada frame del vídeo táctico de FC Barcelona vs Athletic Club como `H`, `A`, `D` o `X`. Se puntúa con `spellF1`, y en caso de empate decide `macroF1`. El baseline oficial a superar es `spellF1` 0.2433 sobre el partido completo. **La entrega cierra el 17 de noviembre de 2026.**

El enunciado oficial está en [RETO.md](RETO.md), [CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md) y [RULES.md](RULES.md). Son copia del repo oficial [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027), configurado aquí como remoto `upstream`.

## Estructura

```
solution/        nuestro código (el comando que regenera la entrega)
submissions/     CSVs que entregamos, <equipo>.csv
docs/            documento técnico (Markdown)
starter/         kit oficial: loaders, baseline y notebooks
evaluation/      scorer oficial, ground truth público (5 min)
games.py         rutas y constantes del partido
data/            datos del partido (NO se versionan)
```

## Puesta en marcha

```bash
pip install -r requirements.txt            # scorer: numpy y pandas
pip install -r requirements-starter.txt    # notebooks: kloppy y matplotlib
python starter/baseline.py                 # regenera los 4 baselines
python evaluation/run_evaluation.py        # puntúa todo lo que hay en evaluation/results/
python evaluation/validate_submission.py submissions/<equipo>.csv
```

## Datos

Los datos los dan los organizadores y **no se pueden redistribuir**, así que `data/` está en `.gitignore`. Descomprime el bundle así:

```
data/tactical/   BAR-ATH_tactical_atd.{txt,xml}, _homography.csv, _smart_tagging.xml, (.mp4)
data/broadcast/  BAR-ATH_broadcast_atd.{txt,xml}, _homography.csv, (.mp4)
```

Antes de hacer push, comprueba que `git status` no muestra nada de `data/`, ni ningún `.mp4`.

## Traer cambios del repo oficial

```bash
git fetch upstream
git diff HEAD upstream/main -- CHALLENGE.md EVALUATION.md evaluation/ starter/
```

## Licencia

Código y documentación bajo licencia MIT (ver [LICENSE](LICENSE)), igual que el kit oficial. Los datos del partido no están cubiertos por esta licencia.
