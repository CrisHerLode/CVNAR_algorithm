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
from calc_metric_function import metrics_from_supports

UMBRAL_DISTANCIA_REGLAS_DEFAULT = 0.05
TOP_RUNS_TO_MERGE_DEFAULT = 3
TOP_FINAL_RULES_DEFAULT = None
OUTPUT_TOP_RULES_TXT_DEFAULT = "top_reglas_finales.txt"
LOG_GLOB_DEFAULT = "run_*.txt"
MIN_LIFT_DEFAULT = 1.0
MIN_SUPPORT_FRAC_DEFAULT = 0.02
MAX_PER_STRUCTURE_TOP_DEFAULT = 2
INTERVAL_FULL_LOW = 0.0
INTERVAL_FULL_HIGH = 1.0
INTERVAL_EPS = 1e-6
# Ranking multiobjetivo de runs (must sum to 1.0)
W_FITNESS = 0.30
W_COVERED = 0.30
W_DIVERSITY = 0.40


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


def is_full_interval(lo, hi):
    return (
        abs(lo - INTERVAL_FULL_LOW) <= INTERVAL_EPS
        and abs(hi - INTERVAL_FULL_HIGH) <= INTERVAL_EPS
    )


def active_interval_pairs(values, attribute_type):
    pairs = []
    n = len(attribute_type) // 2
    for i in range(n):
        lo_idx = i * 2
        hi_idx = lo_idx + 1
        role = attribute_type[lo_idx]
        if role in (1, 2):
            pairs.append((values[lo_idx], values[hi_idx]))
    return pairs


def iter_active_conditions(values, attribute_type):
    """Yield (attr_idx, role, lo, hi) for antecedent/consequent attributes."""
    n = len(attribute_type) // 2
    for i in range(n):
        lo_idx = i * 2
        hi_idx = lo_idx + 1
        role = attribute_type[lo_idx]
        if role in (1, 2):
            yield i, int(role), float(values[lo_idx]), float(values[hi_idx])


def active_values_vector(values, attribute_type):
    """Endpoints of active (ant/cons) intervals only; ignores unused genes."""
    vec = []
    for lo, hi in active_interval_pairs(values, attribute_type):
        vec.extend((lo, hi))
    return vec


def discriminative_signature(values, attribute_type):
    """Canonical key ignoring unused genes and individual [0,1] conditions.

    Rules that only differ by catch-all literals (e.g. adding A0 [0,1]) collapse
    to the same signature and are treated as duplicates.
    """
    parts = []
    for attr_idx, role, lo, hi in iter_active_conditions(values, attribute_type):
        if is_full_interval(lo, hi):
            continue
        parts.append((attr_idx, role, round(lo, 6), round(hi, 6)))
    return frozenset(parts)


def structural_fingerprint(values, attribute_type):
    """Attribute/role pairs of non-[0,1] conditions (interval-agnostic niche)."""
    return frozenset(
        (attr_idx, role)
        for attr_idx, role, lo, hi in iter_active_conditions(values, attribute_type)
        if not is_full_interval(lo, hi)
    )


def count_full_active_conditions(values, attribute_type):
    return sum(
        1
        for _, _, lo, hi in iter_active_conditions(values, attribute_type)
        if is_full_interval(lo, hi)
    )


def count_discriminative_conditions(values, attribute_type):
    return sum(
        1
        for _, _, lo, hi in iter_active_conditions(values, attribute_type)
        if not is_full_interval(lo, hi)
    )


def is_distinct_rule(candidate, selected, threshold):
    """Drop duplicates: same printed rule, same discriminative signature, or close intervals."""
    cand_text = rule_body_to_string(candidate["values"], candidate["attribute_type"])
    cand_sig = discriminative_signature(candidate["values"], candidate["attribute_type"])
    cand_active = active_values_vector(candidate["values"], candidate["attribute_type"])
    for rule in selected:
        if cand_text == rule_body_to_string(rule["values"], rule["attribute_type"]):
            return False
        other_sig = discriminative_signature(rule["values"], rule["attribute_type"])
        if cand_sig and cand_sig == other_sig:
            return False
        if candidate["attribute_type"] == rule["attribute_type"]:
            other_active = active_values_vector(rule["values"], rule["attribute_type"])
            if len(cand_active) == len(other_active) and cand_active:
                if euclidean_distance(cand_active, other_active) < threshold:
                    return False
            elif not cand_active and not other_active:
                return False
    return True


def is_catch_all_rule(values, attribute_type):
    pairs = active_interval_pairs(values, attribute_type)
    if not pairs:
        return True
    return all(is_full_interval(lo, hi) for lo, hi in pairs)


def infer_dataset_size(run_rows, candidate_rules):
    sizes = []
    for row in run_rows:
        if row.get("n_rows") is not None:
            sizes.append(int(row["n_rows"]))
    if sizes:
        return max(sizes)
    for row in run_rows:
        if row.get("covered") is not None:
            sizes.append(int(row["covered"]))
    for rule in candidate_rules:
        for key in ("rule_support", "ant_support", "cons_support"):
            val = rule.get(key)
            if val is not None:
                sizes.append(int(val))
    return max(sizes) if sizes else None


def min_rule_support_required(dataset_size, min_support_frac, min_rule_support_abs):
    if min_rule_support_abs is not None:
        return max(1, int(min_rule_support_abs))
    if dataset_size is None or min_support_frac is None:
        return 1
    return max(2, math.ceil(dataset_size * min_support_frac))


def passes_quality_filter(
    rule,
    *,
    min_lift,
    min_support_frac,
    min_rule_support_abs,
    dataset_size,
    enabled,
):
    if not enabled:
        return True, None

    if is_catch_all_rule(rule["values"], rule["attribute_type"]):
        return False, "catch_all"

    lift = rule.get("lift")
    if lift is None:
        return False, "sin_lift"
    if lift <= min_lift:
        return False, "lift_bajo"

    min_support = min_rule_support_required(dataset_size, min_support_frac, min_rule_support_abs)
    rule_support = rule.get("rule_support")
    if rule_support is None:
        return False, "sin_rule_support"
    if int(rule_support) < min_support:
        return False, "support_bajo"

    return True, None


def rule_body_to_string(values, attribute_types):
    rules = generate_rules([values], [attribute_types])
    return rules[0] if rules else ""


def rule_pretty_line(values, attribute_types):
    body = rule_body_to_string(values, attribute_types)
    if not body:
        return "Rule: (sin intervalos validos)"
    return f"Rule:  {body}"


def enrich_rule_extra_metrics(rule, dataset_size):
    """Add gain/leverage/wracc/conviction derived from supports."""
    extra = {
        "gain": None,
        "leverage": None,
        "wracc": None,
        "conviction": None,
    }
    if (
        dataset_size is not None
        and rule.get("ant_support") is not None
        and rule.get("cons_support") is not None
        and rule.get("rule_support") is not None
    ):
        extra = metrics_from_supports(
            rule["ant_support"],
            rule["cons_support"],
            rule["rule_support"],
            int(dataset_size),
        )
    rule.update(extra)
    return rule


def format_rule_metrics_line(rule):
    def fmt(v, digits=6):
        if v is None:
            return "NA"
        if isinstance(v, float) and math.isinf(v):
            return "inf"
        if isinstance(v, float):
            return f"{v:.{digits}g}"
        return str(v)

    return (
        "metrics: "
        f"ant_sup={rule['ant_support']} | cons_sup={rule['cons_support']} | rule_sup={rule['rule_support']} | "
        f"conf={fmt(rule.get('confidence'))} | lift={fmt(rule.get('lift'))} | "
        f"acc={fmt(rule.get('accuracy'))} | sup={fmt(rule.get('support_metric'))} | "
        f"cf={fmt(rule.get('cf'))} | gain={fmt(rule.get('gain'))} | "
        f"leverage={fmt(rule.get('leverage'))} | wracc={fmt(rule.get('wracc'))} | "
        f"conviction={fmt(rule.get('conviction'))}"
    )


def parse_log_file(path):
    txt = read_log_text(path)
    row = {
        "run": os.path.basename(path),
        "time": None,
        "best": None,
        "covered": None,
        "diversity": None,
        "n_solutions": None,
        "n_rows": None,
        "best_fitness_list": [],
        "intervals_values": [],
        "attribute_type_values": [],
        "ant_support_list": [],
        "cons_support_list": [],
        "rule_support_list": [],
        "confidence_list": [],
        "lift_list": [],
        "accuracy_list": [],
        "support_metric_list": [],
        "cf_list": [],
    }

    m_time = re.search(r"Execution time:\s*([0-9.]+)\s*mins", txt)
    if m_time:
        row["time"] = float(m_time.group(1))

    m_n = re.search(r"\bN:\s*([0-9]+)\s*,\s*M:\s*([0-9]+)", txt)
    if m_n:
        row["n_rows"] = int(m_n.group(1))

    m_best = re.search(r"Best fitness:\s*\[([^\]]+)\]", txt)
    if m_best:
        row["best"] = float(m_best.group(1).split(",")[0].strip())

    m_cov = re.search(r"Covered records number:\s*([0-9]+)", txt)
    if m_cov:
        row["covered"] = int(m_cov.group(1))

    m_nsol = re.search(r"N solutions:\s*([0-9]+)", txt)
    if m_nsol:
        row["n_solutions"] = int(m_nsol.group(1))

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

    ant_support = extract_list_after("Ant support:", txt)
    if isinstance(ant_support, list):
        row["ant_support_list"] = ant_support

    cons_support = extract_list_after("Cons support:", txt)
    if isinstance(cons_support, list):
        row["cons_support_list"] = cons_support

    rule_support = extract_list_after("Rules support:", txt)
    if isinstance(rule_support, list):
        row["rule_support_list"] = rule_support

    confidence = extract_list_after("Confidence metric:", txt)
    if isinstance(confidence, list):
        row["confidence_list"] = confidence

    lift = extract_list_after("Lift metric:", txt)
    if isinstance(lift, list):
        row["lift_list"] = lift

    accuracy = extract_list_after("Accuracy metric:", txt)
    if isinstance(accuracy, list):
        row["accuracy_list"] = accuracy

    support_metric = extract_list_after("Support metric:", txt)
    if isinstance(support_metric, list):
        row["support_metric_list"] = support_metric

    cf = extract_list_after("Certainty Factor metric:", txt)
    if isinstance(cf, list):
        row["cf_list"] = cf

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
        help="Reglas maximas en la salida (por defecto: N solutions detectado en logs)",
    )
    p.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="No filtrar reglas triviales (catch-all), de bajo lift o bajo support",
    )
    p.add_argument(
        "--min-lift",
        type=float,
        default=MIN_LIFT_DEFAULT,
        metavar="L",
        help=f"Lift minimo estricto (por defecto: {MIN_LIFT_DEFAULT})",
    )
    p.add_argument(
        "--min-support-frac",
        type=float,
        default=MIN_SUPPORT_FRAC_DEFAULT,
        metavar="F",
        help=f"Support minimo como fraccion del dataset (por defecto: {MIN_SUPPORT_FRAC_DEFAULT})",
    )
    p.add_argument(
        "--min-rule-support",
        type=int,
        default=None,
        metavar="N",
        help="Support minimo absoluto en filas (anula el calculo por fraccion)",
    )
    p.add_argument(
        "--max-per-structure-top",
        type=int,
        default=MAX_PER_STRUCTURE_TOP_DEFAULT,
        metavar="K",
        help=(
            "Maximo de reglas finales por huella estructural "
            f"(atributos/roles no-[0,1]; por defecto: {MAX_PER_STRUCTURE_TOP_DEFAULT})"
        ),
    )


def execute_summary(sargs: SimpleNamespace) -> int:
    """sargs: logs_dir_abs, log_glob, output_abs, umbral, top_merge, top_rules, quality filter opts."""
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
        r["score"] = W_FITNESS * b + W_COVERED * c + W_DIVERSITY * d

    ranked = sorted(valid, key=lambda x: x["score"], reverse=True)
    best = ranked[0]
    print("\n=== RANKING MULTIOBJETIVO ===")
    print(
        f"Pesos: {W_FITNESS:.2f}*best_fitness + {W_COVERED:.2f}*covered_records + "
        f"{W_DIVERSITY:.2f}*diversity"
    )
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
    quality_filter = not getattr(sargs, "no_quality_filter", False)
    min_lift = getattr(sargs, "min_lift", MIN_LIFT_DEFAULT)
    min_support_frac = getattr(sargs, "min_support_frac", MIN_SUPPORT_FRAC_DEFAULT)
    min_rule_support_abs = getattr(sargs, "min_rule_support", None)

    auto_top_rules = None
    pool_runs = ranked if quality_filter else ranked[:top_merge]
    for r in pool_runs[:top_merge]:
        if r.get("n_solutions") is not None:
            auto_top_rules = max(1, int(r["n_solutions"]))
            break
    if auto_top_rules is None:
        auto_top_rules = 10

    top_rules = max(1, int(sargs.top_rules)) if sargs.top_rules is not None else auto_top_rules

    candidate_rules = []
    for r in pool_runs:
        n = min(len(r["best_fitness_list"]), len(r["intervals_values"]), len(r["attribute_type_values"]))
        for i in range(n):
            candidate_rules.append(
                {
                    "source_run": r["run"],
                    "fitness": r["best_fitness_list"][i],
                    "values": r["intervals_values"][i],
                    "attribute_type": r["attribute_type_values"][i],
                    "ant_support": r["ant_support_list"][i] if i < len(r["ant_support_list"]) else None,
                    "cons_support": r["cons_support_list"][i] if i < len(r["cons_support_list"]) else None,
                    "rule_support": r["rule_support_list"][i] if i < len(r["rule_support_list"]) else None,
                    "confidence": r["confidence_list"][i] if i < len(r["confidence_list"]) else None,
                    "lift": r["lift_list"][i] if i < len(r["lift_list"]) else None,
                    "accuracy": r["accuracy_list"][i] if i < len(r["accuracy_list"]) else None,
                    "support_metric": r["support_metric_list"][i] if i < len(r["support_metric_list"]) else None,
                    "cf": r["cf_list"][i] if i < len(r["cf_list"]) else None,
                }
            )

    dataset_size = infer_dataset_size(run_rows, candidate_rules)
    min_support_required = min_rule_support_required(dataset_size, min_support_frac, min_rule_support_abs)

    candidate_rules.sort(
        key=lambda x: (
            -(x["fitness"] if x["fitness"] is not None else float("-inf")),
            count_full_active_conditions(x["values"], x["attribute_type"]),
            -count_discriminative_conditions(x["values"], x["attribute_type"]),
        )
    )
    selected_rules = []
    rejected_counts = {}
    structure_counts = {}
    umbral = sargs.umbral
    max_per_structure_top = max(1, int(getattr(sargs, "max_per_structure_top", MAX_PER_STRUCTURE_TOP_DEFAULT)))
    for cand in candidate_rules:
        ok, reason = passes_quality_filter(
            cand,
            min_lift=min_lift,
            min_support_frac=min_support_frac,
            min_rule_support_abs=min_rule_support_abs,
            dataset_size=dataset_size,
            enabled=quality_filter,
        )
        if not ok:
            rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
            continue
        if not is_distinct_rule(cand, selected_rules, umbral):
            rejected_counts["duplicada"] = rejected_counts.get("duplicada", 0) + 1
            continue
        fp = structural_fingerprint(cand["values"], cand["attribute_type"])
        if structure_counts.get(fp, 0) >= max_per_structure_top:
            rejected_counts["max_estructura"] = rejected_counts.get("max_estructura", 0) + 1
            continue
        selected_rules.append(cand)
        structure_counts[fp] = structure_counts.get(fp, 0) + 1
        if len(selected_rules) == top_rules:
            break

    print("\n=== TOP REGLAS FINALES (MERGE) ===")
    pool_label = f"all_runs({len(pool_runs)})" if quality_filter else f"top_runs={top_merge}"
    print(
        f"Configuracion: pool={pool_label}, top_rules={top_rules}, "
        f"umbral_distancia={umbral}, max_per_structure_top={max_per_structure_top}, "
        f"top_rules_source={'--top-rules' if sargs.top_rules is not None else 'auto(n_solutions)'}"
    )
    if quality_filter:
        print(
            f"Filtro calidad: lift>{min_lift}, rule_support>={min_support_required} "
            f"(dataset~{dataset_size if dataset_size is not None else 'NA'}, frac={min_support_frac})"
        )
    else:
        print("Filtro calidad: desactivado (--no-quality-filter)")
    if rejected_counts:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(rejected_counts.items()))
        print(f"Reglas descartadas: {parts}")
    if not selected_rules:
        print("No se pudieron construir reglas finales (faltan listas en los logs o filtro muy estricto).")
        if quality_filter:
            print(
                "Sugerencia: prueba --no-quality-filter, baja --min-lift, "
                "reduce --min-support-frac, aumenta --merge-runs o --max-per-structure-top."
            )
        return 0

    for rule in selected_rules:
        enrich_rule_extra_metrics(rule, dataset_size)

    for i, rule in enumerate(selected_rules, start=1):
        print(f" {i}. {rule_pretty_line(rule['values'], rule['attribute_type'])}")
        print(f"    fitness={rule['fitness']:.6f} | run={rule['source_run']}")
        print(f"    {format_rule_metrics_line(rule)}")

    out_path = sargs.output_abs
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    lines = [
        f"# Generado: {datetime.now().isoformat(timespec='seconds')}",
        f"# Logs en: {logs_dir_abs}",
        f"# Glob: {sargs.log_glob}",
        f"# Pesos ranking runs: {W_FITNESS:.2f}*best_fitness_norm + "
        f"{W_COVERED:.2f}*covered_records_norm + {W_DIVERSITY:.2f}*diversity_norm",
        f"# Mejor run: {best['run']} | score={best['score']:.6f}",
        f"# Runs fusionados para reglas ({top_merge}): "
        + ", ".join(r["run"] for r in ranked[:top_merge]),
        f"# Umbral_distancia_duplicadas (texto/firma discriminativa/[0,1] ignorados; "
        f"misma estructura attribute_type en intervalos activos): {umbral}",
        f"# Max reglas por estructura (attrs/roles no-[0,1]): {max_per_structure_top}",
    ]
    if quality_filter:
        lines.append(
            f"# Filtro calidad: lift>{min_lift}, rule_support>={min_support_required}, "
            f"sin catch-all; pool={pool_label}"
        )
    else:
        lines.append("# Filtro calidad: desactivado")
    lines.extend(
        [
        "",
        f"TOTAL_REGLAS_SELECCIONADAS: {len(selected_rules)}",
        "",
        ]
    )
    for i, rule in enumerate(selected_rules, start=1):
        lines.append(f"{i}. {rule_pretty_line(rule['values'], rule['attribute_type'])}")
        lines.append(f"   # fitness={rule['fitness']:.6f} | run={rule['source_run']}")
        lines.append(f"   # {format_rule_metrics_line(rule)}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")
    print(f"\nEscrito: {out_path}")
    return 0


def summary_namespace_from_args(args, logs_dir_abs, output_abs):
    return SimpleNamespace(
        logs_dir_abs=logs_dir_abs,
        log_glob=args.log_glob,
        output_abs=output_abs,
        umbral=args.umbral_distancia,
        top_merge=max(1, args.merge_runs),
        top_rules=(max(1, args.top_rules) if args.top_rules is not None else None),
        no_quality_filter=args.no_quality_filter,
        min_lift=args.min_lift,
        min_support_frac=args.min_support_frac,
        min_rule_support=args.min_rule_support,
        max_per_structure_top=args.max_per_structure_top,
    )


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
        cmd.extend(["--amplitude-penalty", str(args.amplitude_penalty)])
        if args.niching:
            cmd.append("--niching")
            cmd.extend(["--sharing-radius", str(args.sharing_radius)])
            cmd.extend(["--sharing-alpha", str(args.sharing_alpha)])
            cmd.extend(
                ["--genotypic-distance-threshold", str(args.genotypic_distance_threshold)]
            )
            if args.max_per_structure is not None:
                cmd.extend(["--max-per-structure", str(args.max_per_structure)])
        else:
            cmd.append("--no-niching")
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

    summary_ns = summary_namespace_from_args(args, logs_dir_abs, output_abs)
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
    p_batch.add_argument(
        "--niching",
        dest="niching",
        action="store_true",
        default=True,
        help="Activar niching estructural dinamico en CVOA (default: on)",
    )
    p_batch.add_argument(
        "--no-niching",
        dest="niching",
        action="store_false",
        help="Desactivar niching estructural dinamico",
    )
    p_batch.add_argument(
        "--sharing-radius",
        type=float,
        default=0.5,
        metavar="SIGMA",
        help="Radio de fitness sharing (distancia Jaccard estructural, default: 0.5)",
    )
    p_batch.add_argument(
        "--sharing-alpha",
        type=float,
        default=1.0,
        metavar="A",
        help="Exponente de la funcion de sharing (default: 1.0)",
    )
    p_batch.add_argument(
        "--max-per-structure",
        type=int,
        default=None,
        metavar="K",
        help="Maximo de reglas elite con la misma estructura de atributos",
    )
    p_batch.add_argument(
        "--genotypic-distance-threshold",
        type=float,
        default=0.25,
        metavar="D",
        help="Distancia estructural minima para preferir otro donante de infeccion",
    )
    p_batch.add_argument(
        "--amplitude-penalty",
        type=float,
        default=0.35,
        metavar="W",
        help=(
            "Penalizar intervalos activos anchos: fitness *= 1 - W * mean_width "
            "(default: 0.35; 0 desactiva)"
        ),
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
        summary_ns = summary_namespace_from_args(args, logs_dir_abs, output_abs)
        return execute_summary(summary_ns)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
