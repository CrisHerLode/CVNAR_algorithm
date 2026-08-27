"""
Comparativa de metricas entre variantes CVNAR (base / niching / niching-amp)
para fobj1 y fobj2 (tabla estilo VLMOHSNAR).

Salida: #R, Av*, ext_coverage_pct, div_uniq_struct, div_mean_struct_d

  python comparar_variantes.py ./BK --csv ./BK/BK.csv --objf 1 2 \\
      --out-path ./BK/reportes/comparacion_variantes_fobj12
"""

from __future__ import annotations

import argparse
import math
import re
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from calc_metric_function import calcMetric
from niching import structural_distance, structure_fingerprint
from support_function import calcRegCub, calcSupport

VARIANT_CHOICES = ("base", "niching", "niching-amp")
DEFAULT_VARIANTS = VARIANT_CHOICES
# CLI uses hyphen; on-disk folders use underscore for niching-amp.
VARIANT_DIR_SUFFIX = {
    "base": "base",
    "niching": "niching",
    "niching-amp": "niching_amp",
}
PRINTED_KEYS = (
    "conf",
    "lift",
    "acc",
    "cf",
    "gain",
    "leverage",
    "wracc",
    "netconf",
    "yule_q",
    "sup",
)

RULE_RE = re.compile(
    r"^\d+\.\s+Rule:\s+(.*?)\n\s+# fitness=([0-9.]+)\s+\|\s+run=([^\n]+)$",
    re.MULTILINE | re.DOTALL,
)
ATTR_RE = re.compile(r"A(\d+)\s*\[([0-9.]+),([0-9.]+)\]")


def parse_top_rules(path: Path) -> list[dict]:
    txt = path.read_text(encoding="utf-8")
    rules = []
    for m in RULE_RE.finditer(txt):
        rule_str, fit_str, run_name = m.groups()
        ant_raw, cons_raw = [x.strip() for x in rule_str.split("->", 1)]
        ant = [(int(a), float(lo), float(hi)) for a, lo, hi in ATTR_RE.findall(ant_raw)]
        cons = [(int(a), float(lo), float(hi)) for a, lo, hi in ATTR_RE.findall(cons_raw)]
        rules.append(
            {
                "rule": rule_str.strip(),
                "fitness": float(fit_str),
                "run": run_name.strip(),
                "ant": ant,
                "cons": cons,
            }
        )
    return rules


def summarize_rules(rules: list[dict]) -> dict:
    if not rules:
        return {"n": 0}

    fits = [r["fitness"] for r in rules]
    ant_sizes = [len(r["ant"]) for r in rules]
    cons_sizes = [len(r["cons"]) for r in rules]

    ant_counts: Counter = Counter()
    cons_counts: Counter = Counter()
    run_counts: Counter = Counter()

    intervals = []
    for r in rules:
        run_counts[r["run"]] += 1
        for a in r["ant"]:
            ant_counts[a[0]] += 1
            intervals.append((a[1], a[2]))
        for a in r["cons"]:
            cons_counts[a[0]] += 1
            intervals.append((a[1], a[2]))

    full_01 = sum(1 for lo, hi in intervals if lo == 0.0 and hi == 1.0)
    narrow_05 = sum(1 for lo, hi in intervals if (hi - lo) <= 0.50)

    return {
        "n": len(rules),
        "fit_mean": st.mean(fits),
        "fit_min": min(fits),
        "fit_max": max(fits),
        "fit_sd": st.pstdev(fits),
        "ant_avg": st.mean(ant_sizes),
        "cons_avg": st.mean(cons_sizes),
        "runs": dict(run_counts),
        "ant_counts": dict(sorted(ant_counts.items())),
        "cons_counts": dict(sorted(cons_counts.items())),
        "intervals_total": len(intervals),
        "interval_full_01": full_01,
        "interval_full_01_pct": 100.0 * full_01 / len(intervals) if intervals else 0.0,
        "interval_narrow_05": narrow_05,
        "interval_narrow_05_pct": 100.0 * narrow_05 / len(intervals) if intervals else 0.0,
    }


def _normalize_dataframe(df: pd.DataFrame, exclude=None):
    exclude = set(exclude or [])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_scale = [c for c in numeric_cols if c not in exclude]
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    if cols_to_scale:
        df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df_scaled


def encode_rule_for_support(rule: dict, n_cols: int):
    values = []
    attr_types = []
    for _ in range(n_cols):
        values.extend([0.0, 1.0])
        attr_types.extend([0, 0])

    valid = True
    for attr_idx, lo, hi in rule["ant"]:
        if attr_idx >= n_cols:
            valid = False
            continue
        values[attr_idx * 2] = lo
        values[attr_idx * 2 + 1] = hi
        attr_types[attr_idx * 2] = 1
        attr_types[attr_idx * 2 + 1] = 1
    for attr_idx, lo, hi in rule["cons"]:
        if attr_idx >= n_cols:
            valid = False
            continue
        values[attr_idx * 2] = lo
        values[attr_idx * 2 + 1] = hi
        attr_types[attr_idx * 2] = 2
        attr_types[attr_idx * 2 + 1] = 2

    return values, attr_types, valid


def evaluate_rules_on_csv(rules: list[dict], csv_path: Path) -> dict:
    data = pd.read_csv(csv_path, sep=";")
    data = data.fillna(data.mean(numeric_only=True))
    data = _normalize_dataframe(data, exclude=[])

    n_cols = data.shape[1]
    metrics_rows = []
    values_set = []
    types_set = []
    invalid_rules = 0

    for r in rules:
        values, attr_types, valid = encode_rule_for_support(r, n_cols)
        if not valid:
            invalid_rules += 1
        values_set.append(values)
        types_set.append(attr_types)

        supports = calcSupport(data, values, attr_types)
        m = calcMetric(data, supports)
        metrics_rows.append(
            {
                "support_ant": supports[0],
                "support_cons": supports[1],
                "support_rule": supports[2],
                "confidence": m[0],
                "lift": m[1],
                "leverage_norm": m[2],
                "accuracy": m[3],
                "support": m[4],
                "cf": m[5],
                "leverage": m[7],
                "gain": m[9],
                "wracc": m[10],
                "conviction": m[11] if math.isfinite(m[11]) else None,
                "conviction_inf": not math.isfinite(m[11]),
                "netconf": m[12],
                "yule_q": m[13],
            }
        )

    rules_cov = calcRegCub(data, values_set, types_set)
    n_rows = len(data.index)

    def mean_metric(key: str) -> float:
        vals = [row[key] for row in metrics_rows if row.get(key) is not None]
        return st.mean(vals) if vals else 0.0

    finite_conv = [row["conviction"] for row in metrics_rows if row["conviction"] is not None]
    n_conv_inf = sum(1 for row in metrics_rows if row["conviction_inf"])

    return {
        "n_rows": n_rows,
        "invalid_rules": invalid_rules,
        "coverage_records": rules_cov,
        "coverage_pct": 100.0 * rules_cov / n_rows if n_rows else 0.0,
        "mean_confidence": mean_metric("confidence"),
        "mean_lift": mean_metric("lift"),
        "mean_support": mean_metric("support"),
        "mean_accuracy": mean_metric("accuracy"),
        "mean_cf": mean_metric("cf"),
        "mean_gain": mean_metric("gain"),
        "mean_leverage": mean_metric("leverage"),
        "mean_wracc": mean_metric("wracc"),
        "mean_conviction": st.mean(finite_conv) if finite_conv else 0.0,
        "conviction_inf_count": n_conv_inf,
        "mean_netconf": mean_metric("netconf"),
        "mean_yule_q": mean_metric("yule_q"),
        "lift_gt_1": sum(1 for row in metrics_rows if row["lift"] > 1.0),
        "conf_ge_08": sum(1 for row in metrics_rows if row["confidence"] >= 0.8),
    }


def default_run_dir(dataset_dir: Path, objf: str, variant: str) -> Path:
    suffix = VARIANT_DIR_SUFFIX.get(variant, variant.replace("-", "_"))
    return dataset_dir / f"runs_fobj{objf}_{suffix}"


def default_top_name(objf: str) -> str:
    return f"top_reglas_finales_fobj{objf}.txt"


def printed_metric_averages(top_path: Path) -> tuple[dict[str, Optional[float]], int]:
    """Averages of metrics printed in the top-rules file."""
    txt = top_path.read_text(encoding="utf-8")
    vals = {k: [] for k in PRINTED_KEYS}
    for line in txt.splitlines():
        if "metrics:" not in line:
            continue
        for k in PRINTED_KEYS:
            m = re.search(rf"(?<![a-z_]){k}=([0-9.eE+-]+)", line)
            if m:
                vals[k].append(float(m.group(1)))
    avgs = {k: (st.mean(v) if v else None) for k, v in vals.items()}
    return avgs, len(vals["sup"])


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        p = c / float(total)
        if p > 0:
            h -= p * math.log(p, 2)
    return h


def _infer_n_attrs(rules: list[dict], csv_path: Optional[Path]) -> int:
    if csv_path is not None and csv_path.is_file():
        try:
            import pandas as pd

            return int(pd.read_csv(csv_path, sep=";", nrows=0).shape[1])
        except Exception:
            pass
    max_idx = -1
    for r in rules:
        for a, _lo, _hi in r.get("ant", []) + r.get("cons", []):
            max_idx = max(max_idx, int(a))
    return max(max_idx + 1, 1)


def _rule_to_attr_type(rule: dict, n_attrs: int) -> list[int]:
    attr = [0] * (n_attrs * 2)
    for a, _lo, _hi in rule.get("ant", []):
        if 0 <= a < n_attrs:
            attr[a * 2] = 1
            attr[a * 2 + 1] = 1
    for a, _lo, _hi in rule.get("cons", []):
        if 0 <= a < n_attrs:
            attr[a * 2] = 2
            attr[a * 2 + 1] = 2
    return attr


def compute_diversity_stats(
    rules: list[dict],
    csv_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Structural diversity of a top-rules set (Jaccard distance on attr roles)."""
    if not rules:
        return {
            "uniq_text": 0,
            "uniq_struct": 0,
            "max_per_struct": 0,
            "runs_used": 0,
            "run_entropy": 0.0,
            "ant_entropy": 0.0,
            "cons_entropy": 0.0,
            "mean_struct_d": None,
            "min_struct_d": None,
            "max_struct_d": None,
            "frac_pairs_d_ge_05": None,
            "mean_width": None,
        }

    n_attrs = _infer_n_attrs(rules, csv_path)
    attrs = [_rule_to_attr_type(r, n_attrs) for r in rules]
    fps = [structure_fingerprint(a) for a in attrs]
    fp_counts = Counter(fps)

    dists = []
    for i in range(len(attrs)):
        for j in range(i + 1, len(attrs)):
            dists.append(structural_distance(attrs[i], attrs[j]))

    ant_c: Counter = Counter()
    cons_c: Counter = Counter()
    widths = []
    for r in rules:
        for a, lo, hi in r.get("ant", []):
            ant_c[a] += 1
            widths.append(float(hi) - float(lo))
        for a, lo, hi in r.get("cons", []):
            cons_c[a] += 1
            widths.append(float(hi) - float(lo))

    run_c = Counter(r.get("run", "") for r in rules)

    return {
        "uniq_text": len({r.get("rule", "") for r in rules}),
        "uniq_struct": len(set(fps)),
        "max_per_struct": max(fp_counts.values()) if fp_counts else 0,
        "runs_used": len(run_c),
        "run_entropy": _entropy(run_c),
        "ant_entropy": _entropy(ant_c),
        "cons_entropy": _entropy(cons_c),
        "mean_struct_d": st.mean(dists) if dists else None,
        "min_struct_d": min(dists) if dists else None,
        "max_struct_d": max(dists) if dists else None,
        "frac_pairs_d_ge_05": (
            sum(1 for d in dists if d >= 0.5) / float(len(dists)) if dists else None
        ),
        "mean_width": st.mean(widths) if widths else None,
    }


def collect_variant_row(
    label: str,
    top_path: Path,
    csv_path: Optional[Path],
) -> dict[str, Any]:
    rules = parse_top_rules(top_path)
    summary = summarize_rules(rules)
    printed, n_printed = printed_metric_averages(top_path)
    external = evaluate_rules_on_csv(rules, csv_path) if csv_path is not None else None
    diversity = compute_diversity_stats(rules, csv_path=csv_path)
    return {
        "label": label,
        "top_path": top_path,
        "n_rules": summary.get("n", 0),
        "summary": summary,
        "printed": printed,
        "n_printed": n_printed,
        "external": external,
        "diversity": diversity,
        "missing": False,
        "error": None,
    }


def compare_objf_variants(
    dataset_dir: Path,
    objfs: Iterable[str],
    variants: Iterable[str] = DEFAULT_VARIANTS,
    csv_path: Optional[Path] = None,
    run_dir_fn=default_run_dir,
    top_name_fn=default_top_name,
) -> list[dict[str, Any]]:
    """
    Build one row per (objf, variant).

    Skips missing tops (row marked missing=True) so incomplete base runs
    do not abort the whole comparison.
    """
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    csv_resolved = Path(csv_path).expanduser().resolve() if csv_path else None
    rows: list[dict[str, Any]] = []

    for objf in objfs:
        objf_s = str(objf)
        for variant in variants:
            label = f"fobj{objf_s} {variant}"
            run_dir = run_dir_fn(dataset_dir, objf_s, variant)
            top_path = run_dir / top_name_fn(objf_s)
            if not top_path.is_file():
                rows.append(
                    {
                        "label": label,
                        "top_path": top_path,
                        "n_rules": 0,
                        "summary": {},
                        "printed": {},
                        "n_printed": 0,
                        "external": None,
                        "diversity": {},
                        "missing": True,
                        "error": f"missing top file: {top_path}",
                    }
                )
                continue
            try:
                rows.append(collect_variant_row(label, top_path, csv_resolved))
            except Exception as exc:  # noqa: BLE001 - report and continue
                rows.append(
                    {
                        "label": label,
                        "top_path": top_path,
                        "n_rules": 0,
                        "summary": {},
                        "printed": {},
                        "n_printed": 0,
                        "external": None,
                        "diversity": {},
                        "missing": True,
                        "error": str(exc),
                    }
                )
    return rows


def _fmt(v: Optional[float], digits: int = 6) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def format_comparison_report(
    rows: list[dict[str, Any]],
    *,
    dataset_dir: Path,
    csv_path: Optional[Path],
    title: str = "Comparativa variantes CVNAR (estilo VLMOHSNAR)",
) -> str:
    lines: list[str] = []
    lines.append(title)
    lines.append(f"Dataset dir: {dataset_dir}")
    if csv_path:
        lines.append(f"CSV (solo para ext_coverage_pct): {csv_path}")
    lines.append(
        "Columnas: #R + Av* (top) + ext_coverage_pct + div_uniq_struct + div_mean_struct_d"
    )
    lines.append("Variantes: base / niching / niching-amp | fobj: las indicadas")
    lines.append("")

    missing = [r for r in rows if r.get("missing")]
    if missing:
        lines.append("=== Omitidos / incompletos ===")
        for r in missing:
            lines.append(f"- {r['label']}: {r.get('error')}")
        lines.append("")

    present = [r for r in rows if not r.get("missing")]

    lines.append(
        "method | #R | Avconf | Avsup | Avcf | Avlift | Avacc | Avgain | "
        "Avleverage | Avwracc | Avnetconf | AvyuleQ | "
        "ext_coverage_pct | div_uniq_struct | div_mean_struct_d"
    )
    for r in present:
        p = r.get("printed") or {}
        ev = r.get("external") or {}
        d = r.get("diversity") or {}
        lines.append(
            "{label} | {n} | {conf} | {sup} | {cf} | {lift} | {acc} | {gain} | "
            "{lev} | {wr} | {nc} | {yq} | {cov} | {us} | {md}".format(
                label=r["label"],
                n=r.get("n_printed") or r.get("n_rules") or 0,
                conf=_fmt(p.get("conf")),
                sup=_fmt(p.get("sup")),
                cf=_fmt(p.get("cf")),
                lift=_fmt(p.get("lift")),
                acc=_fmt(p.get("acc")),
                gain=_fmt(p.get("gain")),
                lev=_fmt(p.get("leverage")),
                wr=_fmt(p.get("wracc")),
                nc=_fmt(p.get("netconf")),
                yq=_fmt(p.get("yule_q")),
                cov=_fmt(ev.get("coverage_pct"), 2),
                us=d.get("uniq_struct", 0),
                md=_fmt(d.get("mean_struct_d"), 3),
            )
        )
    lines.append("")
    lines.append(
        "Notas: Av* = medias del top; ext_coverage_pct = % filas cubiertas por el set; "
        "div_mean_struct_d = distancia Jaccard media (0=igual, 1=disjunto)."
    )
    return "\n".join(lines).rstrip() + "\n"


def rows_to_table_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flat rows for CSV/Excel: paper Av* + coverage + diversity only."""
    table = []
    for r in rows:
        label = r["label"]
        parts = label.split(" ", 1)
        objf = parts[0].replace("fobj", "") if parts else ""
        variant = parts[1] if len(parts) > 1 else ""
        p = r.get("printed") or {}
        ev = r.get("external") or {}
        d = r.get("diversity") or {}
        table.append(
            {
                "method": label,
                "objf": objf,
                "variant": variant,
                "status": "missing" if r.get("missing") else "ok",
                "n_rules": r.get("n_printed") or r.get("n_rules") or 0,
                "Avconf": p.get("conf"),
                "Avsup": p.get("sup"),
                "Avcf": p.get("cf"),
                "Avlift": p.get("lift"),
                "Avacc": p.get("acc"),
                "Avgain": p.get("gain"),
                "Avleverage": p.get("leverage"),
                "Avwracc": p.get("wracc"),
                "Avnetconf": p.get("netconf"),
                "AvyuleQ": p.get("yule_q"),
                "ext_coverage_pct": ev.get("coverage_pct"),
                "div_uniq_struct": d.get("uniq_struct"),
                "div_mean_struct_d": d.get("mean_struct_d"),
            }
        )
    return table


def write_comparison_csv(rows: list[dict[str, Any]], out_path: Path) -> Path:
    import csv

    table = rows_to_table_dicts(rows)
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(table[0].keys()) if table else [
        "method",
        "objf",
        "variant",
        "status",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in table:
            writer.writerow(row)
    return out_path


def write_comparison_excel(rows: list[dict[str, Any]], out_path: Path) -> Path:
    import pandas as pd

    table = rows_to_table_dicts(rows)
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(table)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="report", index=False)
    return out_path


def write_comparison_report(
    dataset_dir: Path,
    objfs: Iterable[str],
    out_path: Path,
    *,
    variants: Iterable[str] = DEFAULT_VARIANTS,
    csv_path: Optional[Path] = None,
    title: Optional[str] = None,
    formats: Optional[Iterable[str]] = None,
) -> dict[str, Path]:
    """
    Compare + write report files.

    formats: subset of {'txt','csv','xlsx'} (default: all three).
    Returns dict format -> path.
    """
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    fmt_set = {f.lower() for f in (formats or ("txt", "csv", "xlsx"))}
    rows = compare_objf_variants(
        dataset_dir,
        objfs=objfs,
        variants=variants,
        csv_path=csv_path,
    )
    written: dict[str, Path] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")

    if "txt" in fmt_set:
        txt_path = stem.with_suffix(".txt")
        report = format_comparison_report(
            rows,
            dataset_dir=dataset_dir,
            csv_path=Path(csv_path).resolve() if csv_path else None,
            title=title
            or "Comparativa variantes CVNAR (fobj x base/niching/niching-amp)",
        )
        txt_path.write_text(report, encoding="utf-8")
        written["txt"] = txt_path

    if "csv" in fmt_set:
        csv_out = stem.with_suffix(".csv")
        written["csv"] = write_comparison_csv(rows, csv_out)

    if "xlsx" in fmt_set:
        xlsx_out = stem.with_suffix(".xlsx")
        try:
            written["xlsx"] = write_comparison_excel(rows, xlsx_out)
        except PermissionError:
            alt = stem.with_name(stem.name + "_nuevo").with_suffix(".xlsx")
            written["xlsx"] = write_comparison_excel(rows, alt)
            print(
                f"AVISO: {xlsx_out.name} bloqueado (¿abierto en Excel?). "
                f"Escrito en {alt.name}"
            )

    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compara metricas de tops entre variantes "
            "(base / niching / niching-amp) para fobj indicadas."
        )
    )
    p.add_argument(
        "dataset_dir",
        help="Carpeta del dataset (p.ej. ./BK) con subcarpetas runs_fobj{N}_{variant}",
    )
    p.add_argument(
        "--csv",
        default=None,
        help="CSV del dataset para evaluacion externa (default: <dataset_dir>/<nombre>.csv si existe)",
    )
    p.add_argument(
        "--objf",
        nargs="+",
        default=["1", "2"],
        help="Funciones objetivo a incluir (default: 1 2)",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        choices=VARIANT_CHOICES,
        help="Variantes a incluir (default: base niching niching-amp)",
    )
    p.add_argument(
        "--out-path",
        default=None,
        help=(
            "Ruta base de salida (extension indiferente). "
            "Default: <dataset_dir>/reportes/comparacion_variantes_fobj...."
        ),
    )
    p.add_argument(
        "--format",
        dest="formats",
        nargs="+",
        default=["txt", "csv", "xlsx"],
        choices=["txt", "csv", "xlsx"],
        help="Formatos de salida (default: txt csv xlsx)",
    )
    return p


def _guess_csv(dataset_dir: Path) -> Optional[Path]:
    name = dataset_dir.name
    candidate = dataset_dir / f"{name}.csv"
    if candidate.is_file():
        return candidate
    csvs = sorted(dataset_dir.glob("*.csv"))
    return csvs[0] if len(csvs) == 1 else None


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        print(f"No existe carpeta dataset: {dataset_dir}")
        return 2

    csv_path = Path(args.csv).expanduser().resolve() if args.csv else _guess_csv(dataset_dir)
    objfs = [str(x) for x in args.objf]
    variants = list(args.variants)

    if args.out_path:
        out_path = Path(args.out_path).expanduser().resolve()
    else:
        tag = "".join(objfs)
        out_path = dataset_dir / "reportes" / f"comparacion_variantes_fobj{tag}.txt"

    written = write_comparison_report(
        dataset_dir,
        objfs=objfs,
        out_path=out_path,
        variants=variants,
        csv_path=csv_path,
        formats=args.formats,
    )
    if "txt" in written:
        print(written["txt"].read_text(encoding="utf-8"))
    for fmt, path in written.items():
        print(f"Reporte {fmt}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
