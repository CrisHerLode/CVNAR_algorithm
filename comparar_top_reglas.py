"""
Compara dos archivos top10_reglas_finales.txt generados por cvoa_multiobjetivo.

Uso:
  python comparar_top_reglas.py <archivo_o_carpeta_A> <archivo_o_carpeta_B>

Si se pasa carpeta, asume <carpeta>/top10_reglas_finales.txt.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics as st
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from calc_metric_function import calcMetric
from support_function import calcRegCub, calcSupport

RULE_RE = re.compile(
    r"^\d+\.\s+Rule:\s+(.*?)\n\s+# fitness=([0-9.]+)\s+\|\s+run=([^\n]+)$",
    re.MULTILINE | re.DOTALL,
)
ATTR_RE = re.compile(r"A(\d+)\s*\[([0-9.]+),([0-9.]+)\]")


def resolve_top_file(path_like: str, top_file_name: str = "top10_reglas_finales.txt") -> Path:
    p = Path(path_like).expanduser().resolve()
    if p.is_dir():
        p = p / top_file_name
    return p


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

    ant_counts: Counter[int] = Counter()
    cons_counts: Counter[int] = Counter()
    run_counts: Counter[str] = Counter()

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


def _fmt_counts(d: dict) -> str:
    if not d:
        return "-"
    return ", ".join(f"A{k}:{v}" if isinstance(k, int) else f"{k}:{v}" for k, v in d.items())


def print_summary(label: str, src: Path, summary: dict, emit=print) -> None:
    emit(f"## {label}")
    emit(f"- Archivo: {src}")
    emit(f"- Reglas parseadas: {summary['n']}")
    if summary["n"] == 0:
        emit("- No se encontraron reglas con el formato esperado.")
        emit("")
        return
    emit(
        "- Fitness: "
        f"media={summary['fit_mean']:.6f}, min={summary['fit_min']:.6f}, "
        f"max={summary['fit_max']:.6f}, sd={summary['fit_sd']:.6f}"
    )
    emit(f"- Tamano medio regla: ant={summary['ant_avg']:.2f}, cons={summary['cons_avg']:.2f}")
    emit(
        "- Intervalos [0,1]: "
        f"{summary['interval_full_01']}/{summary['intervals_total']} "
        f"({summary['interval_full_01_pct']:.1f}%)"
    )
    emit(
        "- Intervalos estrechos (<=0.5): "
        f"{summary['interval_narrow_05']}/{summary['intervals_total']} "
        f"({summary['interval_narrow_05_pct']:.1f}%)"
    )
    emit(f"- Reparto por run: {_fmt_counts(summary['runs'])}")
    emit(f"- Uso de atributos en antecedentes: {_fmt_counts(summary['ant_counts'])}")
    emit(f"- Uso de atributos en consecuentes: {_fmt_counts(summary['cons_counts'])}")
    emit("")


def normalize_dataframe(df: pd.DataFrame, exclude=None):
    exclude = set(exclude or [])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_scale = [c for c in numeric_cols if c not in exclude]

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    if cols_to_scale:
        df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df_scaled


def encode_rule_for_support(rule: dict, n_cols: int) -> tuple[list[float], list[int], bool]:
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
    data = normalize_dataframe(data, exclude=[])

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


def print_external_eval(label: str, csv_path: Path, eval_stats: dict, emit=print) -> None:
    emit(f"## Evaluacion Externa {label}")
    emit(f"- CSV: {csv_path}")
    emit(f"- Filas evaluadas: {eval_stats['n_rows']}")
    emit(f"- Reglas con atributos fuera de rango del CSV: {eval_stats['invalid_rules']}")
    emit(
        f"- Cobertura del set (al menos una regla dispara): "
        f"{eval_stats['coverage_records']}/{eval_stats['n_rows']} ({eval_stats['coverage_pct']:.2f}%)"
    )
    emit(
        f"- Promedio por regla: conf={eval_stats['mean_confidence']:.4f}, "
        f"lift={eval_stats['mean_lift']:.4f}, support={eval_stats['mean_support']:.4f}, "
        f"accuracy={eval_stats['mean_accuracy']:.4f}, cf={eval_stats['mean_cf']:.4f}"
    )
    emit(
        f"- Metricas adicionales: gain={eval_stats['mean_gain']:.4f}, "
        f"leverage={eval_stats['mean_leverage']:.4f}, wracc={eval_stats['mean_wracc']:.4f}, "
        f"conviction={eval_stats['mean_conviction']:.4f}"
        + (
            f" (inf en {eval_stats['conviction_inf_count']} reglas con conf=1)"
            if eval_stats.get("conviction_inf_count")
            else ""
        )
        + f", netconf={eval_stats['mean_netconf']:.4f}, yule_q={eval_stats['mean_yule_q']:.4f}"
    )
    emit(
        f"- Reglas con lift>1: {eval_stats['lift_gt_1']} | "
        f"Reglas con conf>=0.8: {eval_stats['conf_ge_08']}"
    )
    emit("")


def emit_metric_descriptions(emit=print) -> None:
    emit("## Significado de Metricas")
    emit("- Reglas parseadas: cantidad de reglas leidas del top10_reglas_finales.txt.")
    emit("- Fitness media/min/max/sd: calidad interna segun la funcion objetivo del experimento.")
    emit("- Tamano medio regla (ant/cons): numero medio de condiciones en antecedente y consecuente.")
    emit("- Intervalos [0,1]: porcentaje de intervalos totalmente abiertos (reglas mas generales).")
    emit("- Intervalos estrechos (<=0.5): porcentaje de intervalos mas restrictivos/especificos.")
    emit("- Reparto por run: cuantas reglas finales provienen de cada corrida.")
    emit("- Uso de atributos ant/cons: frecuencia de cada atributo en antecedente/consecuente.")
    emit("- Diferencia (B menos A): valor en B menos valor en A; positivo => B mayor.")
    emit("- Cobertura del set: porcentaje de filas donde al menos una regla dispara.")
    emit("- Confidence: P(consecuente | antecedente), fiabilidad de cada regla.")
    emit("- Lift: ganancia respecto a la tasa base; >1 aporta informacion, ~1 aporta poco.")
    emit("- Support: frecuencia de filas donde antecedente y consecuente se cumplen simultaneamente.")
    emit("- Accuracy: exactitud de la regla como clasificador binario cumplir/no cumplir.")
    emit("- CF (certainty factor): fortaleza de la regla frente a la tasa base del consecuente.")
    emit("- Gain: conf - P(C); >0 indica confianza por encima de la prevalencia del consecuente.")
    emit("- Leverage: P(AC) - P(A)P(C); dependencia positiva si >0.")
    emit("- WRAcc: P(A)*(conf - P(C)); clasico Lavrac (coincide con leverage).")
    emit("- Conviction: (1-P(C))/(1-conf); alto si la regla rara vez falla (inf si conf=1).")
    emit("- Netconf: (P(AC)-P(A)P(C))/(P(A)*(1-P(A))); interes neto (estilo tablas VLMOHSNAR).")
    emit("- Yule's Q: (ad-bc)/(ad+bc) en la tabla 2x2 de la regla; asociacion en [-1,1].")
    emit("")


def emit_solo_diagnosis(summary: dict, ev: dict | None, emit=print) -> None:
    emit("## Diagnostico Automatico")

    if not summary.get("n"):
        emit("- No hay reglas suficientes para diagnostico.")
        emit("")
        return

    if summary["interval_narrow_05_pct"] >= 40:
        emit("- Estructura: reglas relativamente especificas (muchos intervalos estrechos).")
    elif summary["interval_full_01_pct"] >= 40:
        emit("- Estructura: reglas relativamente generales (muchos intervalos [0,1]).")
    else:
        emit("- Estructura: complejidad intermedia entre reglas generales y especificas.")

    if ev is None:
        emit("- Sin CSV no se puede diagnosticar rendimiento externo; solo estructura.")
        emit("")
        return

    if ev["coverage_pct"] >= 80 and ev["mean_lift"] <= 1.2:
        emit(
            "- Rendimiento externo: buena cobertura con lift moderado; "
            "reglas utiles para describir el dataset de forma amplia."
        )
    elif ev["coverage_pct"] < 30 and ev["mean_lift"] > 2.0:
        emit(
            "- Rendimiento externo: baja cobertura pero alto lift; "
            "reglas informativas pero de alcance limitado."
        )
    else:
        emit("- Rendimiento externo: perfil mixto entre cobertura y discriminacion.")

    emit(
        f"- Indicadores clave: cobertura {ev['coverage_pct']:.2f}%, "
        f"lift {ev['mean_lift']:.4f}, conf {ev['mean_confidence']:.4f}, "
        f"support {ev['mean_support']:.4f}, wracc {ev['mean_wracc']:.4f}, "
        f"gain {ev['mean_gain']:.4f}."
    )
    emit(
        f"- Reglas con lift>1: {ev['lift_gt_1']}/{summary['n']} | "
        f"conf>=0.8: {ev['conf_ge_08']}/{summary['n']}."
    )
    emit("")


def emit_auto_diagnosis(sa: dict, sb: dict, eva: dict | None, evb: dict | None, emit=print) -> None:
    emit("## Diagnostico Automatico")

    if not (sa.get("n") and sb.get("n")):
        emit("- No hay reglas suficientes para diagnostico comparativo.")
        emit("")
        return

    # Diagnostico estructural
    narrow_diff = sb["interval_narrow_05_pct"] - sa["interval_narrow_05_pct"]
    ant_diff = sb["ant_avg"] - sa["ant_avg"]
    if narrow_diff <= -20 or ant_diff <= -0.4:
        struct_msg = "B parece mas generalista (menos restrictivo) y A mas especifico."
    elif narrow_diff >= 20 or ant_diff >= 0.4:
        struct_msg = "B parece mas especifico/discriminativo y A mas generalista."
    else:
        struct_msg = "A y B tienen complejidad estructural similar."
    emit(f"- Estructura: {struct_msg}")

    # Diagnostico externo si hay CSV
    if eva is not None and evb is not None:
        cov_diff = evb["coverage_pct"] - eva["coverage_pct"]
        lift_diff = evb["mean_lift"] - eva["mean_lift"]
        acc_diff = evb["mean_accuracy"] - eva["mean_accuracy"]
        conf_diff = evb["mean_confidence"] - eva["mean_confidence"]

        if cov_diff > 10 and lift_diff < -0.3:
            emit(
                "- Rendimiento externo: B cubre mas casos, pero con menor capacidad "
                "discriminativa (lift menor)."
            )
        elif cov_diff < -10 and lift_diff > 0.3:
            emit(
                "- Rendimiento externo: B cubre menos casos, pero con mayor "
                "capacidad discriminativa (lift mayor)."
            )
        else:
            emit(
                "- Rendimiento externo: no hay una dominancia clara; depende de si "
                "priorizas cobertura o discriminacion."
            )

        # Recomendacion automatica simple
        score_a = (eva["coverage_pct"] / 100.0) + eva["mean_lift"] + eva["mean_accuracy"] + eva["mean_confidence"]
        score_b = (evb["coverage_pct"] / 100.0) + evb["mean_lift"] + evb["mean_accuracy"] + evb["mean_confidence"]
        winner = "A" if score_a >= score_b else "B"
        emit(
            f"- Recomendacion balanceada automatica: {winner} "
            "(suma de cobertura+lift+accuracy+confidence, sin pesos adicionales)."
        )
        emit(
            "- Recomendacion por objetivo: si priorizas cobertura elige el de mayor cobertura; "
            "si priorizas reglas informativas elige el de mayor lift."
        )
        emit(
            f"- Diferencias clave externas: cobertura {cov_diff:+.2f} pts, "
            f"lift {lift_diff:+.4f}, accuracy {acc_diff:+.4f}, confidence {conf_diff:+.4f}."
        )
    else:
        emit("- Sin CSV no se puede diagnosticar rendimiento externo; solo estructura.")

    emit("")


def run_solo_report(args) -> int:
    top_name = args.top_file_name_a if args.top_file_name_a else args.top_file_name
    pa = resolve_top_file(args.a, top_name)

    if not pa.is_file():
        print(f"No existe archivo: {pa}")
        return 2

    ra = parse_top_rules(pa)
    sa = summarize_rules(ra)

    report_lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        report_lines.append(line)

    emit("# Reporte individual (modo --solo)")
    emit("")
    print_summary("Reglas", pa, sa, emit=emit)

    dataset_root = Path(pa.parent).resolve().parents[0]
    ev = None
    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        if not csv_path.is_file():
            emit(f"No existe CSV para evaluacion externa: {csv_path}")
            return 2
        dataset_root = csv_path.parent
        ev = evaluate_rules_on_csv(ra, csv_path)
        print_external_eval("Reglas", csv_path, ev, emit=emit)

    emit_metric_descriptions(emit=emit)
    emit_solo_diagnosis(sa, ev, emit=emit)

    out_path = Path(args.out_path).expanduser().resolve() if args.out_path else (dataset_root / args.out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    print(f"\nReporte guardado en: {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compara dos top10_reglas_finales.txt o genera reporte individual (--solo)."
    )
    parser.add_argument("a", help="Archivo o carpeta A (o unico conjunto con --solo)")
    parser.add_argument("b", nargs="?", default=None, help="Archivo o carpeta B (omitir con --solo)")
    parser.add_argument(
        "--solo",
        action="store_true",
        help="Reporte de un solo conjunto de reglas (sin comparacion A vs B)",
    )
    parser.add_argument(
        "--top-file-name",
        default="top10_reglas_finales.txt",
        help="Nombre del fichero de reglas cuando A/B son carpetas.",
    )
    parser.add_argument(
        "--top-file-name-a",
        default=None,
        help="Nombre del fichero de reglas para A cuando A es carpeta (prioriza sobre --top-file-name).",
    )
    parser.add_argument(
        "--top-file-name-b",
        default=None,
        help="Nombre del fichero de reglas para B cuando B es carpeta (prioriza sobre --top-file-name).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV para evaluacion externa de ambos conjuntos (mismo separador ';' que main_cvoa).",
    )
    parser.add_argument(
        "--out-name",
        default="comparacion_top_reglas.txt",
        help="Nombre del TXT de salida (se guarda en la raiz del dataset si no se usa --out-path).",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Ruta completa del TXT de salida. Si se indica, tiene prioridad sobre --out-name.",
    )
    args = parser.parse_args()

    if args.solo:
        return run_solo_report(args)
    if args.b is None:
        parser.error("se requiere el argumento b (o usa --solo para un solo conjunto)")

    top_name_a = args.top_file_name_a if args.top_file_name_a else args.top_file_name
    top_name_b = args.top_file_name_b if args.top_file_name_b else args.top_file_name
    pa = resolve_top_file(args.a, top_name_a)
    pb = resolve_top_file(args.b, top_name_b)

    if not pa.is_file():
        print(f"No existe archivo A: {pa}")
        return 2
    if not pb.is_file():
        print(f"No existe archivo B: {pb}")
        return 2

    ra = parse_top_rules(pa)
    rb = parse_top_rules(pb)
    sa = summarize_rules(ra)
    sb = summarize_rules(rb)

    report_lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        report_lines.append(line)

    print_summary("A", pa, sa, emit=emit)
    print_summary("B", pb, sb, emit=emit)

    if sa["n"] and sb["n"]:
        emit("## Diferencia (B menos A)")
        emit(f"- Diferencia fitness medio: {sb['fit_mean'] - sa['fit_mean']:+.6f}")
        emit(f"- Diferencia ant promedio: {sb['ant_avg'] - sa['ant_avg']:+.2f}")
        emit(f"- Diferencia cons promedio: {sb['cons_avg'] - sa['cons_avg']:+.2f}")
        emit(
            "- Diferencia % intervalos estrechos (<=0.5): "
            f"{sb['interval_narrow_05_pct'] - sa['interval_narrow_05_pct']:+.1f} puntos"
        )

    dataset_root = Path(Path(args.csv).expanduser().resolve().parent) if args.csv else Path(
        Path(pa.parent).resolve().parents[0]
    )

    eva = None
    evb = None
    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        if not csv_path.is_file():
            emit(f"\nNo existe CSV para evaluacion externa: {csv_path}")
            return 2
        dataset_root = csv_path.parent
        eva = evaluate_rules_on_csv(ra, csv_path)
        evb = evaluate_rules_on_csv(rb, csv_path)
        emit("")
        print_external_eval("A", csv_path, eva, emit=emit)
        print_external_eval("B", csv_path, evb, emit=emit)
        emit("## Diferencia Evaluacion Externa (B menos A)")
        emit(f"- Diferencia cobertura del set: {evb['coverage_pct'] - eva['coverage_pct']:+.2f} puntos")
        emit(f"- Diferencia confidence media: {evb['mean_confidence'] - eva['mean_confidence']:+.4f}")
        emit(f"- Diferencia lift medio: {evb['mean_lift'] - eva['mean_lift']:+.4f}")
        emit(f"- Diferencia support medio: {evb['mean_support'] - eva['mean_support']:+.4f}")
        emit(f"- Diferencia accuracy media: {evb['mean_accuracy'] - eva['mean_accuracy']:+.4f}")
        emit(f"- Diferencia CF medio: {evb['mean_cf'] - eva['mean_cf']:+.4f}")
        emit(f"- Diferencia gain medio: {evb['mean_gain'] - eva['mean_gain']:+.4f}")
        emit(f"- Diferencia leverage medio: {evb['mean_leverage'] - eva['mean_leverage']:+.4f}")
        emit(f"- Diferencia WRAcc medio: {evb['mean_wracc'] - eva['mean_wracc']:+.4f}")
        emit(
            f"- Diferencia conviction media (finite): "
            f"{evb['mean_conviction'] - eva['mean_conviction']:+.4f}"
        )
        emit(f"- Diferencia netconf medio: {evb['mean_netconf'] - eva['mean_netconf']:+.4f}")
        emit(f"- Diferencia Yule's Q medio: {evb['mean_yule_q'] - eva['mean_yule_q']:+.4f}")

    emit("")
    emit_metric_descriptions(emit=emit)
    emit_auto_diagnosis(sa, sb, eva, evb, emit=emit)

    out_path = Path(args.out_path).expanduser().resolve() if args.out_path else (dataset_root / args.out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    print(f"\nReporte guardado en: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
