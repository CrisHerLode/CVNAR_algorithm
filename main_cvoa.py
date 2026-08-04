import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cvoa2 import CVOA
import time as time
from support_function import calcSupport
from support_function import calcRegCub
from support_function import generate_rules
from calc_metric_function import calcMetric
from sklearn.preprocessing import MinMaxScaler


def normalize_dataframe(df: pd.DataFrame, exclude=None):
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
    parser.add_argument("objf", help="Objective function: 1 or 2")
    parser.add_argument("plot", help="Output path for fitness curve PNG")
    parser.add_argument(
        "--niching",
        dest="niching",
        action="store_true",
        default=True,
        help="Enable dynamic structural niching (default: on)",
    )
    parser.add_argument(
        "--no-niching",
        dest="niching",
        action="store_false",
        help="Disable dynamic structural niching",
    )
    parser.add_argument(
        "--sharing-radius",
        type=float,
        default=0.5,
        metavar="SIGMA",
        help="Fitness sharing radius on Jaccard structural distance (default: 0.5)",
    )
    parser.add_argument(
        "--sharing-alpha",
        type=float,
        default=1.0,
        metavar="A",
        help="Fitness sharing shape exponent (default: 1.0)",
    )
    parser.add_argument(
        "--max-per-structure",
        type=int,
        default=None,
        metavar="K",
        help="Max elite rules with identical attribute structure (default: n_solutions//4)",
    )
    parser.add_argument(
        "--genotypic-distance-threshold",
        type=float,
        default=0.25,
        metavar="D",
        help="Min structural distance to prefer a different infection donor (default: 0.25)",
    )
    parser.add_argument(
        "--amplitude-penalty",
        type=float,
        default=0.35,
        metavar="W",
        help=(
            "Penalize wide active intervals: fitness *= 1 - W * mean_width "
            "(default: 0.35; use 0 to disable)"
        ),
    )
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)

    data = pd.read_csv(args.csv, sep=";")
    objF = args.objf
    data = data.fillna(data.mean(numeric_only=True))
    data, scaler = normalize_dataframe(data, exclude=[])

    N = data.shape[0]
    M = data.shape[1]
    print(f"N: {N}, M: {M}")
    max_time = int(10 + 10 * np.log10(N * M))
    print(f"Max time: {max_time}")
    n_solutions = int(max(10, np.sqrt(N)))
    print(f"N solutions: {n_solutions}")

    epochs = []
    best_solutions = []
    best_values = []
    best_attributeType = []
    best_fitness = []
    ant_support = []
    cons_support = []
    rule_support = []
    conf_metric = []
    lift_metric = []
    leverage_metric = []
    accuracy_metric = []
    support_metric = []
    cf_metric = []
    cf2_metric = []
    leverage2_metric = []
    accuracy2_metric = []
    gain = []
    wracc_metric = []
    conviction_metric = []

    cvoa = CVOA(
        max_time=max_time,
        data=data,
        n_solutions=n_solutions,
        objF=objF,
        niching=args.niching,
        sharing_radius=args.sharing_radius,
        sharing_alpha=args.sharing_alpha,
        max_per_structure=args.max_per_structure,
        genotypic_distance_threshold=args.genotypic_distance_threshold,
        amplitude_penalty=args.amplitude_penalty,
    )

    time1 = int(round(time.time() * 1000))
    solutions = cvoa.run()
    time2 = int(round(time.time() * 1000)) - time1

    for n in range(len(solutions)):
        epochs.append(n)
        best_solutions.append(solutions[n].kintegers)
        best_values.append(solutions[n].values)
        best_attributeType.append(solutions[n].attributeType)
        best_fitness.append(cvoa.fitness(solutions[n].values, solutions[n].attributeType))
        calculateSupports = calcSupport(data, solutions[n].values, solutions[n].attributeType)
        ant_support.append(calculateSupports[0])
        cons_support.append(calculateSupports[1])
        rule_support.append(calculateSupports[2])
        calculateMetrics = calcMetric(data, calculateSupports)
        conf_metric.append(calculateMetrics[0])
        lift_metric.append(calculateMetrics[1])
        leverage_metric.append(calculateMetrics[2])
        accuracy_metric.append(calculateMetrics[3])
        support_metric.append(calculateMetrics[4])
        cf_metric.append(calculateMetrics[5])
        cf2_metric.append(calculateMetrics[6])
        leverage2_metric.append(calculateMetrics[7])
        accuracy2_metric.append(calculateMetrics[8])
        gain.append(calculateMetrics[9])
        wracc_metric.append(calculateMetrics[10])
        conviction_metric.append(calculateMetrics[11])

    best_fitness_each_Iteration = cvoa.getBestFitnessEachIt()
    mean_fitness_each_Iteration = cvoa.getMeanFitnessEachIt()
    std_fitness_each_Iteration = cvoa.getStdFitnessEachIt()
    calculateRegCov = calcRegCub(data, best_values, best_attributeType)

    print("Execution time: " + str(time2 / 60000) + " mins")
    print("Best solutions: " + str(best_solutions))
    print("Intervals values: " + str(best_values))
    print("Attribute type values: " + str(best_attributeType))
    print("Best fitness: " + str(best_fitness))
    print("Best Fitness for iteration: " + str(best_fitness_each_Iteration))
    print("Mean Fitness for iteration: " + str(mean_fitness_each_Iteration))
    print("Std Fitness for iteration: " + str(std_fitness_each_Iteration))
    print("Ant support: " + str(ant_support))
    print("Cons support: " + str(cons_support))
    print("Rules support: " + str(rule_support))
    print("Confidence metric: " + str(conf_metric))
    print("Lift metric: " + str(lift_metric))
    print("Leverage metric: " + str(leverage_metric))
    print("Leverage metric 2: " + str(leverage2_metric))
    print("Accuracy metric: " + str(accuracy_metric))
    print("Accuracy metric 2: " + str(accuracy2_metric))
    print("Support metric: " + str(support_metric))
    print("Certainty Factor metric: " + str(cf_metric))
    print("Certainty Factor metric 2: " + str(cf2_metric))
    print("Gain: " + str(gain))
    print("WRAcc: " + str(wracc_metric))
    print("Conviction: " + str(conviction_metric))
    print("Covered records number: " + str(calculateRegCov))
    rules = generate_rules(best_values, best_attributeType)
    for rule in rules:
        print("Rule: ", rule)

    plot_fitness_curves(
        best_fitness_each_Iteration,
        mean_fitness_each_Iteration,
        std_fitness_each_Iteration,
        args.plot,
    )


if __name__ == "__main__":
    main()
