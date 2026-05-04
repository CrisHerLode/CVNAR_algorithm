"""
CVOA multiobjetivo: lanzar multiples corridas (main_cvoa.py) y/o resumen
multiobjetivo sobre los logs (ranking, merge de reglas, top10_reglas_finales.txt).

  python cvoa_multiobjetivo.py batch ./lo/LO.csv --out-dir ./lo/runs_umbral --num-runs 30
  python cvoa_multiobjetivo.py resumen ./lo/runs_umbral
"""

from __future__ import annotations

import argparse
import ast
import glob
import math
import os
import re
import statistics as st
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace

from support_function import generate_rules

UMBRAL_DISTANCIA_REGLAS_DEFAULT = 0.08
TOP_RUNS_TO_MERGE_DEFAULT = 3
TOP_FINAL_RULES_DEFAULT = 10
OUTPUT_TOP_RULES_TXT_DEFAULT = "top10_reglas_finales.txt"
LOG_GLOB_DEFAULT = "run_*.txt"


def avg_pairwise_distance(vectors):
    n = len(vectors)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = vectors[i]
            b = vectors[j]
            dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
            total += dist
            pairs += 1
    return total / pairs


def read_log_text(path):
    for enc in ("utf-16", "utf-8"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as fh:
                txt = fh.read()
            return txt.replace("\x00", "")
        except UnicodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read().replace("\x00", "")


def extract_list_after(label, txt):
    idx = txt.find(label)
    if idx == -1:
        return None
    start = txt.find("[", idx)
    if start == -1:
        return None
    depth = 0
    end = None
    for i in range(start, len(txt)):
        ch = txt[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None
    try:
        return ast.literal_eval(txt[start : end + 1])
    except (ValueError, SyntaxError):
        return None


def euclidean_distance(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def is_distinct_rule(candidate, selected, threshold):
    for rule in selected:
        if candidate["attribute_type"] == rule["attribute_type"]:
            if euclidean_distance(candidate["values"], rule["values"]) < threshold:
                return False
    return True


def rule_body_to_string(values, attribute_types):
    rules = generate_rules([values], [attribute_types])
    return rules[0] if rules else ""


def rule_pretty_line(values, attribute_types):
    body = rule_body_to_string(values, attribute_types)
    if not body:
        return "Rule: (sin intervalos validos)"
    return f"Rule:  {body}"


def parse_log_file(path):
    txt = read_log_text(path)
    row = {
        "run": os.path.basename(path),
        "time": None,
        "best": None,
        "covered": None,
        "diversity": None,
        "best_fitness_list": [],
        "intervals_values": [],
        "attribute_type_values": [],
    }

    m_time = re.search(r"Execution time:\s*([0-9.]+)\s*mins", txt)
    if m_time:
        row["time"] = float(m_time.group(1))

    m_best = re.search(r"Best fitness:\s*\[([^\]]+)\]", txt)
    if m_best:
        row["best"] = float(m_best.group(1).split(",")[0].strip())

    m_cov = re.search(r"Covered records number:\s*([0-9]+)", txt)
    if m_cov:
        row["covered"] = int(m_cov.group(1))

    best_fitness_list = extract_list_after("Best fitness:", txt)
    if isinstance(best_fitness_list, list):
        row["best_fitness_list"] = best_fitness_list

    intervals = extract_list_after("Intervals values:", txt)
    if isinstance(intervals, list) and intervals and isinstance(intervals[0], list):
        row["intervals_values"] = intervals
        row["diversity"] = avg_pairwise_distance(intervals)

    attribute_types = extract_list_after("Attribute type values:", txt)
    if isinstance(attribute_types, list) and attribute_types and isinstance(attribute_types[0], list):
        row["attribute_type_values"] = attribute_types

    return row


def resumen(nombre, vals):
    if not vals:
        print(f"{nombre}: sin datos")
        return
    mean = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{nombre}: n={len(vals)}, media={mean:.6f}, sd={sd:.6f}, min={min(vals):.6f}, max={max(vals):.6f}")


def minmax_norm(value, vmin, vmax):
    if value is None:
        return None
    if vmax == vmin:
        return 1.0
    return (value - vmin) / (vmax - vmin)


def add_summary_arguments(p):
    """Opciones del analisis multiobjetivo post-corridas."""
    p.add_argument(
        "--glob",
        dest="log_glob",
        default=LOG_GLOB_DEFAULT,
        metavar="PATTERN",
        help=f"Patron de logs (por defecto: {LOG_GLOB_DEFAULT})",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help=f"TXT de reglas finales (por defecto: carpeta_logs/{OUTPUT_TOP_RULES_TXT_DEFAULT})",
    )
    p.add_argument(
        "--umbral-distancia",
        type=float,
        default=UMBRAL_DISTANCIA_REGLAS_DEFAULT,
        metavar="EPS",
        help="Umbral euclidiano para deduplicar reglas",
    )
    p.add_argument(
        "--merge-runs",
        type=int,
        default=TOP_RUNS_TO_MERGE_DEFAULT,
        metavar="K",
        help="Runs a fusionar en el pool",
    )
    p.add_argument(
        "--top-rules",
        type=int,
        default=TOP_FINAL_RULES_DEFAULT,
        metavar="N",
        help="Reglas maximas en la salida",
    )


def execute_summary(sargs: SimpleNamespace) -> int:
    """sargs: logs_dir_abs, log_glob, output_abs, umbral, top_merge, top_rules."""
    logs_dir_abs = sargs.logs_dir_abs
    if not os.path.isdir(logs_dir_abs):
        print(
            f"No existe la carpeta de logs: {logs_dir_abs}",
            file=sys.stderr,
        )
        if os.path.isfile(logs_dir_abs):
            print(
                "Es un archivo: indica la carpeta con run_*.txt.",
                file=sys.stderr,
            )
        return 2

    pattern = os.path.join(logs_dir_abs, sargs.log_glob)
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No se encontraron logs: {pattern}", file=sys.stderr)
        print("Verifica la carpeta y --glob.", file=sys.stderr)
        return 1

    run_rows = []
    exec_times = []
    best_top = []
    covered = []
    diversity = []

    for f in files:
        row = parse_log_file(f)
        run_rows.append(row)
        if row["time"] is not None:
            exec_times.append(row["time"])
        if row["best"] is not None:
            best_top.append(row["best"])
        if row["covered"] is not None:
            covered.append(row["covered"])
        if row["diversity"] is not None:
            diversity.append(row["diversity"])

    print("=== RESUMEN RUNS ===")
    print(f"Dataset (carpeta de logs): {logs_dir_abs}")
    print(f"Logs encontrados: {len(files)} ({sargs.log_glob})")
    resumen("Execution time (mins)", exec_times)
    resumen("Best fitness top-1", best_top)
    resumen("Covered records", covered)
    resumen("Diversity (avg pairwise distance)", diversity)

    valid = [r for r in run_rows if r["best"] is not None and r["covered"] is not None and r["diversity"] is not None]
    if not valid:
        print("No se pudo calcular ranking: faltan metricas por run.")
        return 0

    bmin, bmax = min(r["best"] for r in valid), max(r["best"] for r in valid)
    cmin, cmax = min(r["covered"] for r in valid), max(r["covered"] for r in valid)
    dmin, dmax = min(r["diversity"] for r in valid), max(r["diversity"] for r in valid)

    for r in valid:
        b = minmax_norm(r["best"], bmin, bmax)
        c = minmax_norm(r["covered"], cmin, cmax)
        d = minmax_norm(r["diversity"], dmin, dmax)
        r["score"] = 0.40 * b + 0.30 * c + 0.30 * d

    ranked = sorted(valid, key=lambda x: x["score"], reverse=True)
    best = ranked[0]
    print("\n=== RANKING MULTIOBJETIVO ===")
    print("Pesos: 0.40*best_fitness + 0.30*covered_records + 0.30*diversity")
    print(
        f"Mejor run: {best['run']} | score={best['score']:.6f} | "
        f"best={best['best']:.6f} | covered={best['covered']} | diversity={best['diversity']:.6f}"
    )
    print("Top 3 runs:")
    for i, r in enumerate(ranked[:3], start=1):
        print(
            f" {i}. {r['run']} | score={r['score']:.6f} | "
            f"best={r['best']:.6f} | covered={r['covered']} | diversity={r['diversity']:.6f}"
        )

    top_merge = max(1, sargs.top_merge)
    top_rules = max(1, sargs.top_rules)

    candidate_rules = []
    for r in ranked[:top_merge]:
        n = min(len(r["best_fitness_list"]), len(r["intervals_values"]), len(r["attribute_type_values"]))
        for i in range(n):
            candidate_rules.append(
                {
                    "source_run": r["run"],
                    "fitness": r["best_fitness_list"][i],
                    "values": r["intervals_values"][i],
                    "attribute_type": r["attribute_type_values"][i],
                }
            )

    candidate_rules.sort(key=lambda x: x["fitness"], reverse=True)
    selected_rules = []
    umbral = sargs.umbral
    for cand in candidate_rules:
        if is_distinct_rule(cand, selected_rules, umbral):
            selected_rules.append(cand)
        if len(selected_rules) == top_rules:
            break

    print("\n=== TOP REGLAS FINALES (MERGE) ===")
    print(
        f"Configuracion: top_runs={top_merge}, top_rules={top_rules}, "
        f"umbral_distancia={umbral}"
    )
    if not selected_rules:
        print("No se pudieron construir reglas finales (faltan listas en los logs).")
        return 0

    for i, rule in enumerate(selected_rules, start=1):
        print(f" {i}. {rule_pretty_line(rule['values'], rule['attribute_type'])}")
        print(f"    fitness={rule['fitness']:.6f} | run={rule['source_run']}")

    out_path = sargs.output_abs
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    lines = [
        f"# Generado: {datetime.now().isoformat(timespec='seconds')}",
        f"# Logs en: {logs_dir_abs}",
        f"# Glob: {sargs.log_glob}",
        "# Pesos ranking runs: 0.40*best_fitness_norm + 0.30*covered_records_norm + 0.30*diversity_norm",
        f"# Mejor run: {best['run']} | score={best['score']:.6f}",
        f"# Runs fusionados para reglas ({top_merge}): "
        + ", ".join(r["run"] for r in ranked[:top_merge]),
        f"# Umbral_distancia_duplicadas (solo si misma estructura attribute_type): {umbral}",
        "",
        f"TOTAL_REGLAS_SELECCIONADAS: {len(selected_rules)}",
        "",
    ]
    for i, rule in enumerate(selected_rules, start=1):
        lines.append(f"{i}. {rule_pretty_line(rule['values'], rule['attribute_type'])}")
        lines.append(f"   # fitness={rule['fitness']:.6f} | run={rule['source_run']}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")
    print(f"\nEscrito: {out_path}")
    return 0


def run_batch(args: argparse.Namespace) -> int:
    csv_abs = os.path.abspath(args.csv)
    if not os.path.isfile(csv_abs):
        print(f"No existe el CSV: {csv_abs}", file=sys.stderr)
        return 2

    out_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir
        else os.path.join(os.path.dirname(csv_abs), "runs_umbral")
    )
    os.makedirs(out_dir, exist_ok=True)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    main_cvoa = os.path.join(repo_root, "main_cvoa.py")
    if not os.path.isfile(main_cvoa):
        print(f"No se encontro main_cvoa.py en: {main_cvoa}", file=sys.stderr)
        return 2

    objf = str(args.objf)
    n = max(1, int(args.num_runs))

    print(f"CSV: {csv_abs}")
    print(f"Salida (logs + graficas): {out_dir}")
    print(f"Carreras: {n} | funcion objetivo (objf): {objf}")
    print(f"Ejecutable: {sys.executable} -u {main_cvoa}")

    failures = 0
    for i in range(1, n + 1):
        plot = os.path.join(out_dir, f"fitness_run_{i}.png")
        log = os.path.join(out_dir, f"run_{i}.txt")
        print(f"\n=== Run {i}/{n} ===")

        cmd = [sys.executable, "-u", main_cvoa, csv_abs, objf, plot]
        with open(log, "w", encoding="utf-8", newline="\n") as logf:
            proc = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=repo_root,
            )
        if proc.returncode != 0:
            print(f"Advertencia: run {i} fallo (codigo {proc.returncode}). Revisa {log}")
            failures += 1
        else:
            print(f"OK -> {log}")

    print(f"\nTerminado. Resultados en: {out_dir}" + (f" ({failures} fallidas)" if failures else ""))

    if args.no_summary:
        return min(1, failures)

    logs_dir_abs = out_dir
    output_abs = (
        os.path.abspath(args.output) if args.output else os.path.join(logs_dir_abs, OUTPUT_TOP_RULES_TXT_DEFAULT)
    )

    summary_ns = SimpleNamespace(
        logs_dir_abs=logs_dir_abs,
        log_glob=args.log_glob,
        output_abs=output_abs,
        umbral=args.umbral_distancia,
        top_merge=max(1, args.merge_runs),
        top_rules=max(1, args.top_rules),
    )
    print("\n=== Resumen multiobjetivo (post-batch) ===\n")
    rc = execute_summary(summary_ns)
    return rc if rc != 0 else min(1, failures)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "CVOA multiobjetivo: subcomando `batch` (N corridas + resumen opcional) o "
            "`resumen` (solo sobre logs)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_batch = sub.add_parser(
        "batch",
        help="Ejecutar N veces main_cvoa.py sobre un CSV y, por defecto, resumen multiobjetivo.",
    )
    p_batch.add_argument("csv", help="Path al archivo CSV (; como separador, como main_cvoa)")
    p_batch.add_argument(
        "--out-dir",
        default=None,
        help=f"Carpeta para run_* y fitness_run_*.png (por defecto: <carpeta del csv>/runs_umbral)",
    )
    p_batch.add_argument("--num-runs", type=int, default=30, metavar="N", help="Numero de corridas")
    p_batch.add_argument("--objf", default="1", help="Argumento objf de main_cvoa (default: 1)")
    p_batch.add_argument(
        "--no-summary",
        action="store_true",
        help="No ejecutar el resumen/reglas despues del batch",
    )
    add_summary_arguments(p_batch)

    p_res = sub.add_parser(
        "resumen",
        help="Solo resumen estadistico y top reglas desde una carpeta de logs existente.",
    )
    p_res.add_argument(
        "dataset",
        metavar="DATASET",
        help="Carpeta con run_*.txt (salida tipica del batch)",
    )
    add_summary_arguments(p_res)

    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "batch":
        return run_batch(args)

    if args.cmd == "resumen":
        logs_dir_abs = os.path.abspath(args.dataset)
        output_abs = (
            os.path.abspath(args.output)
            if args.output
            else os.path.join(logs_dir_abs, OUTPUT_TOP_RULES_TXT_DEFAULT)
        )
        summary_ns = SimpleNamespace(
            logs_dir_abs=logs_dir_abs,
            log_glob=args.log_glob,
            output_abs=output_abs,
            umbral=args.umbral_distancia,
            top_merge=max(1, args.merge_runs),
            top_rules=max(1, args.top_rules),
        )
        return execute_summary(summary_ns)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
