"""
Tops CVNAR: post-filtro minsup y/o top DEFINITIVO por carpeta.

Top definitivo (secuencia exacta sobre candidatas de ESA carpeta):
  1) Robustez:  support >= 0.05
  2) Calidad:   netconf >= 0.25
  3) Diversidad: huella estructural unica + distancia Jaccard minima
  4) Ranking:   netconf desc, luego lift desc

Salidas:
  <runs_dir>/top_reglas_postfilter_minsup0.05.txt
  <runs_dir>/top_reglas_definitivo_sup0.05_nc0.25.txt

Uso:
  python postfilter_hibrido_bk.py --dataset BK --definitive-only
  python postfilter_hibrido_bk.py --dataset BL --definitive-only
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import cvoa_multiobjetivo as mo
from comparar_top_reglas import parse_top_rules
from comparar_variantes import collect_variant_row, printed_metric_averages
from niching import structural_distance


ROOT = Path(__file__).resolve().parent

VARIANT_DIRS = ("base", "niching", "niching_amp")
OBJFS = ("1", "2")


def run_resumen(logs_dir: Path, out_path: Path, min_support_frac: float) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "cvoa_multiobjetivo.py"),
        "resumen",
        str(logs_dir),
        "-o",
        str(out_path),
        "--min-support-frac",
        str(min_support_frac),
    ]
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.call(cmd)


def postfilter_out_name(min_support_frac: float) -> str:
    return f"top_reglas_postfilter_minsup{min_support_frac:.2f}.txt"


def definitive_out_name(min_support_frac: float, min_netconf: float) -> str:
    return f"top_reglas_definitivo_sup{min_support_frac:.2f}_nc{min_netconf:.2f}.txt"


def discover_variant_dirs(dataset: Path) -> list[tuple[str, Path]]:
    found = []
    for objf in OBJFS:
        for variant in VARIANT_DIRS:
            name = f"fobj{objf}_{variant}"
            logs_dir = dataset / f"runs_{name}"
            if logs_dir.is_dir() and list(logs_dir.glob("run_*.txt")):
                found.append((name, logs_dir))
            else:
                print(f"AVISO: omitido (sin runs): {logs_dir}")
    return found


def infer_n_rows(csv_path: Path) -> int:
    import pandas as pd

    return int(len(pd.read_csv(csv_path, sep=";")))


def load_candidates_from_runs(logs_dir: Path) -> list[dict]:
    pattern = str(logs_dir / "run_*.txt")
    files = sorted(glob.glob(pattern))
    candidates = []
    for f in files:
        r = mo.parse_log_file(f)
        n = min(
            len(r["best_fitness_list"]),
            len(r["intervals_values"]),
            len(r["attribute_type_values"]),
        )
        for i in range(n):
            candidates.append(
                {
                    "source_run": r["run"],
                    "source_dir": logs_dir.name,
                    "fitness": r["best_fitness_list"][i],
                    "values": r["intervals_values"][i],
                    "attribute_type": r["attribute_type_values"][i],
                    "ant_support": r["ant_support_list"][i]
                    if i < len(r["ant_support_list"])
                    else None,
                    "cons_support": r["cons_support_list"][i]
                    if i < len(r["cons_support_list"])
                    else None,
                    "rule_support": r["rule_support_list"][i]
                    if i < len(r["rule_support_list"])
                    else None,
                    "confidence": r["confidence_list"][i]
                    if i < len(r["confidence_list"])
                    else None,
                    "lift": r["lift_list"][i] if i < len(r["lift_list"]) else None,
                    "accuracy": r["accuracy_list"][i]
                    if i < len(r["accuracy_list"])
                    else None,
                    "support_metric": r["support_metric_list"][i]
                    if i < len(r["support_metric_list"])
                    else None,
                    "cf": r["cf_list"][i] if i < len(r["cf_list"]) else None,
                }
            )
    return candidates


def select_definitive_top(
    pool_dirs: list[Path],
    *,
    out_path: Path,
    dataset_size: int,
    min_support_frac: float = 0.05,
    min_netconf: float = 0.25,
    min_lift: float = 1.0,
    min_struct_d: float = 0.5,
    interval_umbral: float = 0.05,
    top_rules: int = 10,
) -> list[dict]:
    """
    Secuencia definitiva:
      1) support >= frac
      2) netconf >= min_netconf
      3) diversidad estructural estricta
      4) ranking: netconf desc, lift desc
    """
    candidates: list[dict] = []
    for d in pool_dirs:
        candidates.extend(load_candidates_from_runs(d))

    for cand in candidates:
        mo.enrich_rule_extra_metrics(cand, dataset_size)

    min_support = mo.min_rule_support_required(dataset_size, min_support_frac, None)

    def sort_key(x):
        nc = x.get("netconf")
        lift = x.get("lift")
        return (
            -(nc if nc is not None else float("-inf")),
            -(lift if lift is not None else float("-inf")),
            mo.count_full_active_conditions(x["values"], x["attribute_type"]),
            -mo.count_discriminative_conditions(x["values"], x["attribute_type"]),
        )

    # 4) Ranking primero: luego se recorre en ese orden aplicando filtros 1-3
    candidates.sort(key=sort_key)

    selected: list[dict] = []
    rejected: dict[str, int] = {}
    seen_fps: set = set()

    for cand in candidates:
        # 1) Robustez (+ lift/catch-all)
        ok, reason = mo.passes_quality_filter(
            cand,
            min_lift=min_lift,
            min_support_frac=min_support_frac,
            min_rule_support_abs=None,
            dataset_size=dataset_size,
            enabled=True,
        )
        if not ok:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue

        # 2) Interes/calidad
        nc = cand.get("netconf")
        if nc is None or float(nc) < min_netconf:
            rejected["netconf_bajo"] = rejected.get("netconf_bajo", 0) + 1
            continue

        # 3) Diversidad estructural estricta
        fp = mo.structural_fingerprint(cand["values"], cand["attribute_type"])
        if fp in seen_fps:
            rejected["estructura_duplicada"] = rejected.get("estructura_duplicada", 0) + 1
            continue

        too_close = False
        for prev in selected:
            d = structural_distance(cand["attribute_type"], prev["attribute_type"])
            if d < min_struct_d:
                too_close = True
                break
        if too_close:
            rejected["struct_d_baja"] = rejected.get("struct_d_baja", 0) + 1
            continue

        if not mo.is_distinct_rule(cand, selected, interval_umbral):
            rejected["duplicada_intervalos"] = rejected.get("duplicada_intervalos", 0) + 1
            continue

        selected.append(cand)
        seen_fps.add(fp)
        if len(selected) >= top_rules:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TOP DEFINITIVO CVNAR (por carpeta)",
        f"# Pool: {', '.join(d.name for d in pool_dirs)}",
        "# Secuencia:",
        (
            f"#   1) Robustez: support>={min_support} "
            f"(frac={min_support_frac}, N={dataset_size}), lift>{min_lift}, sin catch-all"
        ),
        f"#   2) Calidad: netconf>={min_netconf}",
        (
            f"#   3) Diversidad: huella estructural unica + Jaccard d>={min_struct_d} "
            f"+ dedup intervalos umbral={interval_umbral}"
        ),
        "#   4) Ranking: netconf desc, luego lift desc",
        f"# top_rules={top_rules}",
        (
            "# Rechazos: "
            + (", ".join(f"{k}={v}" for k, v in sorted(rejected.items())) or "(ninguno)")
        ),
        "",
        f"TOTAL_REGLAS_SELECCIONADAS: {len(selected)}",
        "",
    ]

    for i, rule in enumerate(selected, start=1):
        lines.append(
            mo.rule_pretty_line(rule["values"], rule["attribute_type"]).replace(
                "Rule:", f"{i}. Rule:", 1
            )
        )
        fit = rule.get("fitness")
        fit_s = f"{fit:.6f}" if fit is not None else "NA"
        lines.append(
            f"   # fitness={fit_s} | run={rule['source_run']} | pool={rule['source_dir']}"
        )
        lines.append(f"   # {mo.format_rule_metrics_line(rule)}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Definitivo escrito: {out_path} ({len(selected)} reglas)")
    print(f"  rechazos: {rejected}")
    return selected


def summarize_top(label: str, top_path: Path, csv_path: Path) -> None:
    if not top_path.is_file():
        print(f"- {label}: FALTA {top_path}")
        return
    n = len(parse_top_rules(top_path))
    if n == 0:
        txt = top_path.read_text(encoding="utf-8")
        if "TOTAL_REGLAS_SELECCIONADAS: 0" in txt:
            print(f"- {label}: #R=0 (vacio)")
            return
    printed, n_printed = printed_metric_averages(top_path)
    row = collect_variant_row(label, top_path, csv_path=csv_path)
    ev = row.get("external") or {}
    d = row.get("diversity") or {}

    def f(x, dig=3):
        return "—" if x is None else f"{x:.{dig}f}"

    print(
        f"- {label}: #R={n_printed or n}  "
        f"Avsup={f(printed.get('sup'))} Avnc={f(printed.get('netconf'))} "
        f"Avlift={f(printed.get('lift'))} cov%={f(ev.get('coverage_pct'), 1)} "
        f"uniq={d.get('uniq_struct')} md={f(d.get('mean_struct_d'))}"
    )


def write_definitive_per_folder(
    dataset: Path,
    *,
    out_dir: Path,
    dataset_size: int,
    min_support_frac: float,
    min_netconf: float,
    min_struct_d: float,
    csv_path: Path,
) -> list[Path]:
    written: list[Path] = []
    print(
        f"\n=== TOP DEFINITIVO por carpeta "
        f"(sup>={min_support_frac:.2f}, nc>={min_netconf:.2f}, "
        f"struct_d>={min_struct_d:.2f}) ==="
    )
    for name, logs_dir in discover_variant_dirs(dataset):
        fname = definitive_out_name(min_support_frac, min_netconf)
        local_out = logs_dir / fname
        select_definitive_top(
            [logs_dir],
            out_path=local_out,
            dataset_size=dataset_size,
            min_support_frac=min_support_frac,
            min_netconf=min_netconf,
            min_struct_d=min_struct_d,
        )
        if not local_out.is_file():
            print(f"AVISO: no se genero para {name}")
            continue
        report_out = (
            out_dir
            / f"top_{name}_definitivo_sup{min_support_frac:.2f}_nc{min_netconf:.2f}.txt"
        )
        report_out.write_text(local_out.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(local_out)
        written.append(report_out)
        print(f"OK: {local_out}")
        summarize_top(f"{name} definitivo", local_out, csv_path)
    return written


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Post-filtro minsup y/o TOP DEFINITIVO por carpeta "
            "(support + netconf + diversidad estructural + rank netconf/lift)."
        )
    )
    p.add_argument("--dataset", default="BK")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--fracs", nargs="+", type=float, default=[0.05])
    p.add_argument(
        "--definitive",
        action="store_true",
        help="Generar top definitivo por carpeta",
    )
    p.add_argument(
        "--definitive-only",
        action="store_true",
        help="Solo top definitivo (no regenerar postfilter minsup)",
    )
    p.add_argument("--hybrid", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--hybrid-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--min-netconf", type=float, default=0.25)
    p.add_argument("--hybrid-support-frac", type=float, default=0.05)
    p.add_argument(
        "--min-struct-d",
        type=float,
        default=0.5,
        help="Distancia Jaccard estructural minima entre reglas del top (default 0.5)",
    )
    args = p.parse_args()

    do_definitive = (
        args.definitive or args.definitive_only or args.hybrid or args.hybrid_only
    )
    do_postfilter = not (args.definitive_only or args.hybrid_only)

    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    out_dir = (
        Path(args.out_dir) if args.out_dir else dataset / "reportes" / "postfilter_hibrido"
    )
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dataset / f"{dataset.name}.csv"
    if not csv_path.is_file():
        print(f"No se encuentra CSV: {csv_path}", file=sys.stderr)
        return 1

    variants = discover_variant_dirs(dataset)
    if not variants:
        print("No hay carpetas runs_fobj* con logs.", file=sys.stderr)
        return 1

    fracs = tuple(args.fracs)
    written: list[Path] = []

    if do_postfilter:
        print(f"=== POST-FILTRO min-support-frac {list(fracs)} sobre {dataset.name} ===")
        for name, logs_dir in variants:
            for frac in fracs:
                local_out = logs_dir / postfilter_out_name(frac)
                report_out = out_dir / f"top_{name}_minsup{frac:.2f}.txt"
                rc = run_resumen(logs_dir, local_out, frac)
                if rc != 0:
                    print(f"AVISO: resumen fallo ({rc}) para {local_out}")
                    continue
                if local_out.is_file():
                    report_out.write_text(
                        local_out.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                    written.append(local_out)
                    print(f"OK: {local_out.name}  (+ copia {report_out.name})")
                else:
                    print(f"AVISO: sin reglas / sin fichero para {name} minsup={frac:.2f}")

    if do_definitive:
        n_rows = infer_n_rows(csv_path)
        written.extend(
            write_definitive_per_folder(
                dataset,
                out_dir=out_dir,
                dataset_size=n_rows,
                min_support_frac=args.hybrid_support_frac,
                min_netconf=args.min_netconf,
                min_struct_d=args.min_struct_d,
                csv_path=csv_path,
            )
        )

    if do_postfilter:
        print("\n=== COMPARATIVA RAPIDA (postfilter) ===")
        for name, logs_dir in variants:
            objf = name.split("_", 1)[0].replace("fobj", "")
            orig = logs_dir / f"top_reglas_finales_fobj{objf}.txt"
            summarize_top(f"{name} (orig)", orig, csv_path)
            for frac in fracs:
                path = logs_dir / postfilter_out_name(frac)
                summarize_top(f"{name} postfilter minsup={frac:.2f}", path, csv_path)

    print(f"\nCopias en: {out_dir}")
    print(f"Ficheros generados/actualizados: {len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
