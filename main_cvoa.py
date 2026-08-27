import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from calc_metric_function import calcMetric
from cvoa2 import CVOA
from support_function import calcRegCub, calcSupport, generate_rules

VARIANT_CHOICES = ("base", "niching", "niching-amp")
DEFAULT_VARIANT = "niching-amp"

_NICHING_DEFAULTS = {
    "sharing_radius": 0.5,
    "sharing_alpha": 1.0,
    "max_per_structure": None,
    "genotypic_distance_threshold": 0.25,
}

_AMP_DEFAULTS = {
    "amplitude_support_low": 0.10,
    "amplitude_support_high": 0.80,
    "amplitude_width_power": 2.0,
}

# Per-variant overrides on top of shared niching/amp defaults.
_VARIANT_OVERRIDES = {
    "base": {
        "niching": False,
        "amplitude_penalty": 0.0,
    },
    "niching": {
        "niching": True,
        "amplitude_penalty": 0.0,
    },
    # niching-amp: best amp schedule (adaptive W, width_power=2)
    "niching-amp": {
        "niching": True,
        "amplitude_penalty": 0.35,
    },
}


def resolve_variant(name):
    """Map --variant to fixed CVOA kwargs (base / niching / niching-amp)."""
    key = str(name).strip().lower()
    if key not in VARIANT_CHOICES:
        raise ValueError(
            "Unknown variant {!r}. Choose one of: {}".format(
                name, ", ".join(VARIANT_CHOICES)
            )
        )
    return {
        "variant": key,
        **_NICHING_DEFAULTS,
        **_AMP_DEFAULTS,
        **_VARIANT_OVERRIDES[key],
    }


def variant_help_text():
    return (
        "Algorithm variant: "
        "base (no niching, no amp) | "
        "niching (niching only) | "
        "niching-amp (niching + adaptive amp power=2; default)"
    )


def normalize_dataframe(df, exclude=None):
    exclude = set(exclude or [])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_scale = [c for c in numeric_cols if c not in exclude]
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    if cols_to_scale:
        df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df_scaled, scaler


def plot_fitness_curves(best, mean, std, output_path):
    min_len = min(len(best), len(mean), len(std))
    if min_len == 0:
        print("Nothing to plot: empty curves.")
        return
    x = range(min_len)
    best = best[:min_len]
    mean = mean[:min_len]
    std = std[:min_len]
    fig, ax = plt.subplots()
    ax.set_ylabel("Fitness")
    ax.set_title("Iteration")
    ax.plot(x, best, label="Best Fitness")
    ax.plot(x, mean, label="Mean Fitness")
    ax.plot(x, std, label="Std Fitness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a single CVOA search for numerical association rules."
    )
    parser.add_argument("csv", help="Path to dataset CSV (; separator)")
    parser.add_argument("objf", help="Objective function: 1, 2, or 3")
    parser.add_argument("plot", help="Output path for fitness curve PNG")
    parser.add_argument(
        "--variant",
        choices=VARIANT_CHOICES,
        default=DEFAULT_VARIANT,
        help=variant_help_text(),
    )
    return parser


def load_and_prepare_data(csv_path):
    data = pd.read_csv(csv_path, sep=";")
    data = data.fillna(data.mean(numeric_only=True))
    data, scaler = normalize_dataframe(data, exclude=[])
    return data, scaler


def compute_run_params(n_rows, n_cols):
    max_time = int(10 + 10 * np.log10(n_rows * n_cols))
    n_solutions = int(max(10, np.sqrt(n_rows)))
    return max_time, n_solutions


def run_cvoa(data, objf, cfg, max_time, n_solutions):
    cvoa = CVOA(
        max_time=max_time,
        data=data,
        n_solutions=n_solutions,
        objF=objf,
        niching=cfg["niching"],
        sharing_radius=cfg["sharing_radius"],
        sharing_alpha=cfg["sharing_alpha"],
        max_per_structure=cfg["max_per_structure"],
        genotypic_distance_threshold=cfg["genotypic_distance_threshold"],
        amplitude_penalty=cfg["amplitude_penalty"],
        amplitude_support_low=cfg["amplitude_support_low"],
        amplitude_support_high=cfg["amplitude_support_high"],
        amplitude_width_power=cfg["amplitude_width_power"],
    )
    t0 = int(round(time.time() * 1000))
    solutions = cvoa.run()
    elapsed_ms = int(round(time.time() * 1000)) - t0
    return cvoa, solutions, elapsed_ms


def evaluate_solutions(data, cvoa, solutions):
    """Collect per-solution metrics (same order/fields as the legacy main)."""
    result = {
        "epochs": [],
        "best_solutions": [],
        "best_values": [],
        "best_attributeType": [],
        "best_fitness": [],
        "ant_support": [],
        "cons_support": [],
        "rule_support": [],
        "conf_metric": [],
        "lift_metric": [],
        "leverage_metric": [],
        "accuracy_metric": [],
        "support_metric": [],
        "cf_metric": [],
        "cf2_metric": [],
        "leverage2_metric": [],
        "accuracy2_metric": [],
        "gain": [],
        "wracc_metric": [],
        "conviction_metric": [],
        "netconf_metric": [],
        "yule_q_metric": [],
    }

    for n, sol in enumerate(solutions):
        result["epochs"].append(n)
        result["best_solutions"].append(sol.kintegers)
        result["best_values"].append(sol.values)
        result["best_attributeType"].append(sol.attributeType)
        result["best_fitness"].append(cvoa.fitness(sol.values, sol.attributeType))

        supports = calcSupport(data, sol.values, sol.attributeType)
        result["ant_support"].append(supports[0])
        result["cons_support"].append(supports[1])
        result["rule_support"].append(supports[2])

        metrics = calcMetric(data, supports)
        result["conf_metric"].append(metrics[0])
        result["lift_metric"].append(metrics[1])
        result["leverage_metric"].append(metrics[2])
        result["accuracy_metric"].append(metrics[3])
        result["support_metric"].append(metrics[4])
        result["cf_metric"].append(metrics[5])
        result["cf2_metric"].append(metrics[6])
        result["leverage2_metric"].append(metrics[7])
        result["accuracy2_metric"].append(metrics[8])
        result["gain"].append(metrics[9])
        result["wracc_metric"].append(metrics[10])
        result["conviction_metric"].append(metrics[11])
        result["netconf_metric"].append(metrics[12])
        result["yule_q_metric"].append(metrics[13])

    result["best_fitness_each_Iteration"] = cvoa.getBestFitnessEachIt()
    result["mean_fitness_each_Iteration"] = cvoa.getMeanFitnessEachIt()
    result["std_fitness_each_Iteration"] = cvoa.getStdFitnessEachIt()
    result["covered_records"] = calcRegCub(
        data, result["best_values"], result["best_attributeType"]
    )
    return result


def print_run_report(elapsed_ms, metrics):
    """Print run summary with labels required by cvoa_multiobjetivo.parse_log_file."""
    print("Execution time: " + str(elapsed_ms / 60000) + " mins")
    print("Best solutions: " + str(metrics["best_solutions"]))
    print("Intervals values: " + str(metrics["best_values"]))
    print("Attribute type values: " + str(metrics["best_attributeType"]))
    print("Best fitness: " + str(metrics["best_fitness"]))
    print("Best Fitness for iteration: " + str(metrics["best_fitness_each_Iteration"]))
    print("Mean Fitness for iteration: " + str(metrics["mean_fitness_each_Iteration"]))
    print("Std Fitness for iteration: " + str(metrics["std_fitness_each_Iteration"]))
    print("Ant support: " + str(metrics["ant_support"]))
    print("Cons support: " + str(metrics["cons_support"]))
    print("Rules support: " + str(metrics["rule_support"]))
    print("Confidence metric: " + str(metrics["conf_metric"]))
    print("Lift metric: " + str(metrics["lift_metric"]))
    print("Leverage metric: " + str(metrics["leverage_metric"]))
    print("Leverage metric 2: " + str(metrics["leverage2_metric"]))
    print("Accuracy metric: " + str(metrics["accuracy_metric"]))
    print("Accuracy metric 2: " + str(metrics["accuracy2_metric"]))
    print("Support metric: " + str(metrics["support_metric"]))
    print("Certainty Factor metric: " + str(metrics["cf_metric"]))
    print("Certainty Factor metric 2: " + str(metrics["cf2_metric"]))
    print("Gain: " + str(metrics["gain"]))
    print("WRAcc: " + str(metrics["wracc_metric"]))
    print("Conviction: " + str(metrics["conviction_metric"]))
    print("Netconf: " + str(metrics["netconf_metric"]))
    print("Yule Q: " + str(metrics["yule_q_metric"]))
    print("Covered records number: " + str(metrics["covered_records"]))
    for rule in generate_rules(metrics["best_values"], metrics["best_attributeType"]):
        print("Rule: ", rule)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)

    cfg = resolve_variant(args.variant)
    print(f"Variant: {cfg['variant']}")

    data, _scaler = load_and_prepare_data(args.csv)
    n_rows, n_cols = data.shape[0], data.shape[1]
    print(f"N: {n_rows}, M: {n_cols}")

    max_time, n_solutions = compute_run_params(n_rows, n_cols)
    print(f"Max time: {max_time}")
    print(f"N solutions: {n_solutions}")

    cvoa, solutions, elapsed_ms = run_cvoa(
        data, args.objf, cfg, max_time, n_solutions
    )
    metrics = evaluate_solutions(data, cvoa, solutions)
    print_run_report(elapsed_ms, metrics)

    plot_fitness_curves(
        metrics["best_fitness_each_Iteration"],
        metrics["mean_fitness_each_Iteration"],
        metrics["std_fitness_each_Iteration"],
        args.plot,
    )


if __name__ == "__main__":
    main()
