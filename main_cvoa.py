import sys
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cvoa2 import CVOA
import time as time
from support_function import calcSupport
from support_function import calcRegCub
from support_function import generate_rules
from calc_metric_function import calcMetric
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def normalize_dataframe(df: pd.DataFrame, exclude=None):
    """
    Normaliza a [0,1] solo las columnas numéricas no excluidas.
    exclude: lista de columnas a NO escalar (p. ej., la etiqueta).
    """
    exclude = set(exclude or [])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_scale = [c for c in numeric_cols if c not in exclude]

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    if cols_to_scale:
        df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df_scaled, scaler

def plot_fitness_curves(best, mean, std, output_path):
    """
    Dibuja y guarda las curvas de fitness asegurando mismas longitudes.
    """
    min_len = min(len(best), len(mean), len(std))
    if min_len == 0:
        print("Nothing to plot: empty curves.")
        return

    x = range(min_len)
    best = best[:min_len]
    mean = mean[:min_len]
    std = std[:min_len]

    fig, ax = plt.subplots()
    ax.set_ylabel('Fitness')
    ax.set_title('Iteration')
    ax.plot(x, best, label='Best Fitness')
    ax.plot(x, mean, label='Mean Fitness')
    ax.plot(x, std, label='Std Fitness')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def main():
    if len(sys.argv) < 3:
        print("Error: the argument number must be three")
        sys.exit()

    data = pd.read_csv(sys.argv[1], sep = ';')
    objF = sys.argv[2]

    # Rellenar nulos con la media
    data = data.fillna(data.mean(numeric_only=True))
   
     # Normalizar tras cargar el CSV
    label_cols = []  # ej.: ['class'] si el dataset tiene etiqueta
    data, scaler = normalize_dataframe(data, exclude=label_cols)

    # Calculamos un limite dinamico usando una escala logaritmica para el numero de iteraciones
    N = data.shape[0]
    M = data.shape[1]
    print(f"N: {N}, M: {M}")
    max_time = int(10 + 10 * np.log10(N*M))

    print(f"Max time: {max_time}")

    # Contenedores de resultados
    epochs=[]
    iterations=[]
    best_solutions=[]
    best_values=[]
    best_attributeType=[]
    best_fitness=[]
    ant_support=[]
    cons_support=[]
    rule_support=[]
    conf_metric=[]
    lift_metric=[]
    leverage_metric=[]
    accuracy_metric=[]
    support_metric=[] 
    cf_metric=[]
    cf2_metric=[]
    leverage2_metric=[]
    accuracy2_metric=[]
    gain=[]
    best_fitness_each_Iteration=[]
    mean_fitness_each_Iteration=[]
    std_fitness_each_Iteration=[]

    cvoa = CVOA(max_time = max_time, data = data, n_solutions=100, objF = objF)

    time1 = int(round(time.time() * 1000))
    solutions = cvoa.run() 
    time2 = int(round(time.time() * 1000)) - time1

    for n in range(len(solutions)):
        epochs.append(n)
        best_solutions.append(solutions[n].kintegers)
        best_values.append(solutions[n].values)
        best_attributeType.append(solutions[n].attributeType)
        best_fitness.append(cvoa.fitness(solutions[n].values,solutions[n].attributeType))
        calculateSupports = calcSupport(data, solutions[n].values,solutions[n].attributeType)
        ant_support.append(calculateSupports[0])
        cons_support.append(calculateSupports[1])
        rule_support.append(calculateSupports[2])
        calculateMetrics = calcMetric(data,calculateSupports)
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

    iterations = range(1,max_time+1)
    best_fitness_each_Iteration = cvoa.getBestFitnessEachIt()
    mean_fitness_each_Iteration = cvoa.getMeanFitnessEachIt()
    std_fitness_each_Iteration = cvoa.getStdFitnessEachIt()
    calculateRegCov = calcRegCub(data, best_values, best_attributeType)
    #avg_bestfitness_distance = cvoa.getAvgBestFitnessDist()

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
    print("Covered records number: " + str(calculateRegCov))
    rules = generate_rules(best_values,best_attributeType)
    for rule in rules:
        print("Rule: ", rule)
    #print("Average best fitness distance: " + str(avg_bestfitness_distance))

    # Generar gráfica
    plot_fitness_curves(
        best_fitness_each_Iteration,
        mean_fitness_each_Iteration,
        std_fitness_each_Iteration,
        sys.argv[3]
    )

if __name__ == "__main__":
    main()
