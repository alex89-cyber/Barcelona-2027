# Documento técnico · Barcelona 2027 (borrador v1)

> Borrador vivo. Se completará antes del 17 de noviembre de 2026.

## Enfoque

Tres piezas, todas algorítmicas y sin etiquetas humanas del partido:

1. **Features por frame** (`solution/features.py`) a partir del tracking táctico: forma de cada
   equipo en su dirección de ataque (centroide, amplitud, profundidad, jugadores en campo rival),
   velocidades (media, hacia portería rival), y, cuando hay balón, su posición, velocidad,
   si está fuera del campo y la distancia al jugador más cercano de cada equipo. La dirección de
   ataque se infiere por mitad a partir de la posición media de cada equipo.
2. **Clasificador auto-supervisado** (`solution/model.py`). Las pseudo-etiquetas salen de reglas
   geométricas:
   - `H`/`A`: tramos de ≥10 frames en los que el balón es visible, un jugador de ese equipo está a
     menos de 1,2 m y ningún rival a menos de 2 m, con el juego vivo (velocidad media alta).
   - `D`: balón fuera del campo, o los 3 s previos al inicio de un `SET PIECES` del Smart Tagging.

   Un LightGBM de 3 clases aprende a predecirlas **solo con la forma y velocidad de los equipos**
   (sin el balón), para poder generalizar al 41% de frames sin balón. Se entrena en una mitad y
   predice la otra, para no memorizar sus propias pseudo-etiquetas.
3. **Decodificación temporal** con Viterbi de 3 estados (H, A, D). La emisión combina el
   log-probabilidad del clasificador con bonificaciones por control claro del balón, balón fuera y
   Smart Tagging. Cambiar de posesión o entrar/salir de balón parado tiene un coste fijo, lo que
   evita fragmentar posesiones (lo que más penaliza `spellF1`).

## Datos usados

| activo | uso |
|---|---|
| ATD táctico | todo el modelo |
| Smart Tagging | `SET PIECES` como evidencia de balón parado |
| homografía, vídeo, ATD broadcast | no usados en v1 |

## Datos de entrenamiento

Ninguno etiquetado por humanos. Solo pseudo-etiquetas generadas por reglas sobre el tracking
automático. El ground truth público (5 min) solo se usa para comprobar que el pipeline está bien
alineado, no para entrenar ni ajustar.

## Hiperparámetros

Fijados a priori, no ajustados contra el ground truth público (`solution/run.py`, `PARAMS`).

## Reproducción

```bash
pip install -r requirements-solution.txt
python -m solution.run                    # -> submissions/barcelona2027.csv
python evaluation/validate_submission.py submissions/barcelona2027.csv
```

El resultado es determinista: dos ejecuciones dan el mismo fichero byte a byte.

## Limitaciones y siguientes pasos

- Detección de balón parado sin balón visible (saques de banda, faltas).
- Lógica de "último equipo que controló" para balones sueltos y disputados.
- Usar el ATD broadcast para recuperar el balón cuando el táctico lo pierde.
