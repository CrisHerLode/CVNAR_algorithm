# CVNAR

Optimization algorithm based on coronavirus dynamics for numerical association rules mining.

## Multiobjective workflow (recommended)

Main entrypoint: `cvoa_multiobjetivo.py`

### 1) Run multiple stochastic executions (batch)

```bash
python cvoa_multiobjetivo.py batch ./PATH_DATASET/NAME_DATASET.csv --out-dir ./PATH_OUTPUT/runs_umbral --num-runs 30 --objf 1
```

- `--objf`: objective function (`1` or `2`)
- `--num-runs`: number of runs
- `--out-dir`: folder for `run_*.txt`, `fitness_run_*.png`, and final merged rules

### 2) Summarize existing runs and generate final top rules

```bash
python cvoa_multiobjetivo.py resumen ./PATH_OUTPUT/runs_umbral
```

This creates/updates:

- `./PATH_OUTPUT/runs_umbral/top10_reglas_finales.txt`

### 3) Compare two top-rule sets (e.g., objf1 vs objf2)

```bash
python comparar_top_reglas.py ./PATH_A/runs_umbral ./PATH_B/runs_umbral_fobj2 --csv ./PATH_DATASET/NAME_DATASET.csv --out-path ./PATH_REPORTS/comparacion_top_reglas.txt
```

The comparison report includes:

- structural metrics of rules
- external evaluation (`coverage`, `confidence`, `lift`, `support`, `accuracy`, `cf`)
- automatic diagnosis and metric glossary

## Direct single run (legacy)

```bash
python -u main_cvoa.py ./PATH_DATASET/NAME_DATASET.csv 1 ./PATH_OUTPUT/fitness_run_1.png > ./PATH_OUTPUT/run_1.txt
```

