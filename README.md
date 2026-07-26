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
- `--umbral-distancia EPS`: dedup threshold for same `attribute_type` structure (default: `0.05`)
- Ranking weights for runs: `0.30*fitness + 0.30*coverage + 0.40*diversity`
- `--output FILE`: output filename/path for final rules file
- **Quality filter (on by default):** drops catch-all rules (`[0,1]` intervals), rules with `lift <= 1`, and rules with low support
- `--min-lift L`: minimum lift (default: `1.0`, strict `>`)
- `--min-support-frac F`: minimum support as dataset fraction (default: `0.02`, at least 2 rows)
- `--min-rule-support N`: absolute minimum support in rows (overrides fraction)
- `--no-quality-filter`: disable quality filtering (legacy behaviour)

Generated top-rules file includes, for each rule:

- rule text (`Rule: A... -> A...`)
- origin run and fitness
- per-rule metrics (`ant_sup`, `cons_sup`, `rule_sup`, `conf`, `lift`, `acc`, `sup`, `cf`,
  `gain`, `leverage`, `wracc`, `conviction`)

### 3) Compare two top-rule sets (e.g., objf1 vs objf2)

```bash
python comparar_top_reglas.py ./PATH_A/runs_umbral ./PATH_B/runs_umbral_fobj2 \
  --top-file-name-a top_reglas_finales_fobj1.txt \
  --top-file-name-b top_reglas_finales_fobj2.txt \
  --csv ./PATH_DATASET/NAME_DATASET.csv \
  --out-path ./PATH_REPORTS/comparacion_top_reglas.txt
```

### 3b) Single top-rule report (e.g., only objf2 when objf1 has no valid rules)

```bash
python comparar_top_reglas.py ./PATH_OUTPUT/runs_umbral_fobj2 --solo \
  --top-file-name-a top_reglas_finales_fobj2.txt \
  --csv ./PATH_DATASET/NAME_DATASET.csv \
  --out-path ./PATH_REPORTS/reporte_top_reglas_fobj2.txt
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

