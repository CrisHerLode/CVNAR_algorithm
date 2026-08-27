# CVNAR

Optimization algorithm based on coronavirus dynamics for numerical association rules mining.

## Multiobjective workflow (recommended)

Main entrypoint: `cvoa_multiobjetivo.py`

### 1) Run multiple stochastic executions (batch)

```bash
python cvoa_multiobjetivo.py batch ./PATH_DATASET/NAME_DATASET.csv --out-dir ./PATH_OUTPUT/runs --num-runs 30 --objf 2 --variant niching-amp
```

- `--objf`: objective function (`1`, `2`, or `3`)
  - `1` / `2`: paper objectives (unchanged)
  - `3`: `support + conf + netconf` (coverage + interest; optional experiments)
- `--num-runs`: number of runs
- `--out-dir`: folder for `run_*.txt`, `fitness_run_*.png`, and merged top rules
- `--variant`: algorithm mode (only these three):
  - `base` — paper CVOA (no niching, no amplitude)
  - `niching` — structural niching only
  - `niching-amp` — niching + adaptive amplitude (`W=0.35`, support schedule 0.10→0.80, `width_power=2`; **default**)
- optional: `--no-summary` to skip automatic summary after batch

Single run:

```bash
python main_cvoa.py ./PATH_DATASET/NAME_DATASET.csv 2 ./out/fitness.png --variant niching-amp
```

### 2) Summarize existing runs and generate final top rules

```bash
python cvoa_multiobjetivo.py resumen ./PATH_OUTPUT/runs_umbral --output ./PATH_OUTPUT/top_reglas_finales_fobj1.txt
```

Useful options:

- `--merge-runs K`: how many top runs are merged
- `--top-rules N`: final number of rules (default: detected automatically from `N solutions` in logs)
- `--umbral-distancia EPS`: dedup threshold on active intervals for same `attribute_type` (default: `0.05`); exact printed duplicates and rules that only differ by `[0,1]` literals are always dropped
- `--max-per-structure-top K`: max final rules sharing the same non-`[0,1]` attribute/role fingerprint (default: `2`)
- Ranking prefers higher fitness, then fewer catch-all `[0,1]` conditions
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
  `gain`, `leverage`, `wracc`, `conviction`, `netconf`, `yule_q`)

### 3) Compare two top-rule sets (e.g., objf1 vs objf2)

```bash
python comparar_top_reglas.py ./PATH_A/runs_umbral ./PATH_B/runs_umbral_fobj2 \
  --top-file-name-a top_reglas_finales_fobj1.txt \
  --top-file-name-b top_reglas_finales_fobj2.txt \
  --csv ./PATH_DATASET/NAME_DATASET.csv \
  --out-path ./PATH_REPORTS/comparacion_top_reglas.txt
```

### 3c) Matrix compare: fobj × variants (base / niching / niching-amp)

```bash
python comparar_variantes.py ./BK --csv ./BK/BK.csv --objf 1 2 \
  --out-path ./BK/reportes/comparacion_variantes_fobj12.txt
```

Writes by default `.txt`, `.csv` (`;`) and `.xlsx` (sheet `report`).
Columns: `#R` / `Av*` / `ext_coverage_pct` / `div_uniq_struct` / `div_mean_struct_d`
for fobj 1–2 × base / niching / niching-amp (estilo tabla VLMOHSNAR).
Missing variants are listed and skipped.

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

