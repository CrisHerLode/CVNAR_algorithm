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
- `--out-dir`: folder for `run_*.txt`, `fitness_run_*.png`, and merged top rules
- optional: `--no-summary` to skip automatic summary after batch

### 2) Summarize existing runs and generate final top rules

```bash
python cvoa_multiobjetivo.py resumen ./PATH_OUTPUT/runs_umbral --output ./PATH_OUTPUT/top_reglas_finales_fobj1.txt
```

Useful options:

- `--merge-runs K`: how many top runs are merged
- `--top-rules N`: final number of rules (default: detected automatically from `N solutions` in logs)
- `--umbral-distancia EPS`: dedup threshold for same `attribute_type` structure
- `--output FILE`: output filename/path for final rules file

Generated top-rules file includes, for each rule:

- rule text (`Rule: A... -> A...`)
- origin run and fitness
- per-rule metrics (`ant_sup`, `cons_sup`, `rule_sup`, `conf`, `lift`, `acc`, `sup`, `cf`)

### 3) Compare two top-rule sets (e.g., objf1 vs objf2)

```bash
python comparar_top_reglas.py ./PATH_A/runs_umbral ./PATH_B/runs_umbral_fobj2 \
  --top-file-name-a top_reglas_finales_fobj1.txt \
  --top-file-name-b top_reglas_finales_fobj2.txt \
  --csv ./PATH_DATASET/NAME_DATASET.csv \
  --out-path ./PATH_REPORTS/comparacion_top_reglas.txt
```

Comparison report includes:

- structural metrics of rules
- external evaluation (`coverage`, `confidence`, `lift`, `support`, `accuracy`, `cf`)
- automatic diagnosis
- metric glossary (`Significado de Metricas`)

Output options:

- `--out-path`: full output path (recommended)
- `--out-name`: filename in dataset root (used when `--out-path` is not provided)

## Direct single run (legacy)

```bash
python -u main_cvoa.py ./PATH_DATASET/NAME_DATASET.csv 1 ./PATH_OUTPUT/fitness_run_1.png > ./PATH_OUTPUT/run_1.txt
```

