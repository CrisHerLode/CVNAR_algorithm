# CVNAR

Optimization algorithm based on coronavirus dynamics for numerical association rules mining.

## Multiobjective workflow (recommended)

Main entrypoint: `cvoa_multiobjetivo.py`

### 1) Run multiple stochastic executions (batch)

```bash
python cvoa_multiobjetivo.py batch ./PATH_DATASET/NAME_DATASET.csv --out-dir ./PATH_OUTPUT/runs --num-runs 30 --objf 2 --variant niching-amp
```

After the runs, by default writes **`top_reglas_finales_fobjN.txt`** in that same runs folder
(definitive protocol: support ≥ 0.05 → netconf ≥ 0.25 → structural diversity).

```bash
# Also write classic base and/or minsup tops in the same folder
python cvoa_multiobjetivo.py batch ./PATH_DATASET/NAME_DATASET.csv --out-dir ./PATH_OUTPUT/runs --num-runs 30 --objf 2 --variant niching-amp --base --minsup
```

- `--objf`: objective function (`1`, `2`, or `3`)
  - `1` / `2`: paper objectives (unchanged)
  - `3`: `support + conf + netconf` (coverage + interest; optional experiments)
- `--num-runs`: number of runs
- `--out-dir`: folder for `run_*.txt`, `fitness_run_*.png`, and top rules
- `--variant`: algorithm mode (only these three):
  - `base` — paper CVOA (no niching, no amplitude)
  - `niching` — structural niching only
  - `niching-amp` — niching + adaptive amplitude (`W=0.35`, support schedule 0.10→0.80, `width_power=2`; **default**)
- `--base` / `--minsup`: also write `top_reglas_base_fobjN.txt` / `top_reglas_minsup_fobjN.txt`
- `--no-definitive`: skip finales (only base/minsup if requested)
- `--no-summary`: skip all post-run tops

Single run:

```bash
python main_cvoa.py ./PATH_DATASET/NAME_DATASET.csv 2 ./out/fitness.png --variant niching-amp
```

### 2) Regenerar tops sobre una carpeta de runs (`resumen`)

Same tops as the post-batch step, without re-running CVOA. Works on **one** `runs_*` folder only:

```bash
# Default: top_reglas_finales_fobjN.txt
python cvoa_multiobjetivo.py resumen ./PATH_OUTPUT/runs_fobj2_niching

# Also base and/or minsup
python cvoa_multiobjetivo.py resumen ./PATH_OUTPUT/runs_fobj2_niching --base --minsup

# Only classic base (skip finales)
python cvoa_multiobjetivo.py resumen ./PATH_OUTPUT/runs_fobj2_niching --no-definitive --base -o ./PATH_OUTPUT/top_reglas_base_fobj2.txt
```

- Infers `fobj` from the folder name (`runs_fobj2_...`)
- CSV for finales: parent/`<parent>.csv`, or `--dataset-csv PATH`
- Summary knobs for `--base` / `--minsup`: `--merge-runs`, `--top-rules`, `--min-support-frac`, etc.

Per-folder outputs:

- `top_reglas_finales_fobjN.txt` (default)
- `top_reglas_base_fobjN.txt` (`--base`)
- `top_reglas_minsup_fobjN.txt` (`--minsup`)

### 3) Matrix compare: fobj × variants → Excel/CSV/TXT

```bash
python comparar_variantes.py ./BK --csv ./BK/BK.csv --objf 1 2 \
  --out-path ./BK/reportes/comparacion_variantes_fobj12
```

Writes by default `.txt`, `.csv` (`;`) and `.xlsx` (sheet `report`).
Columns: `#R` / `Av*` / `ext_coverage_pct` / `div_uniq_struct` / `div_mean_struct_d`
for fobj 1–2 × base / niching / niching-amp (estilo tabla VLMOHSNAR).
Uses each folder's `top_reglas_finales_fobjN.txt` by default.
Missing variants are listed and skipped.
