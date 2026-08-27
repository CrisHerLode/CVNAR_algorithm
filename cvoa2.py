from copy import deepcopy

import numpy as np
import random
import sys

from individual import Individual
from niching import (
    apply_fitness_sharing,
    count_structure_in_archive,
    structure_fingerprint,
    structural_distance,
    worst_in_structure,
)


class CVOA:
    MIN_SPREAD = 0
    MAX_SPREAD = 5
    MIN_SUPERSPREAD = 6
    MAX_SUPERSPREAD = 15
    SOCIAL_DISTANCING = 7
    P_ISOLATION = 0.7
    P_TRAVEL = 0.1
    P_REINFECTION = 0.001
    SUPERSPREADER_PERC = 0.1
    DEATH_PERC = 0.06

    def __init__(
        self,
        max_time,
        data,
        n_solutions,
        objF,
        *,
        niching=True,
        sharing_radius=0.5,
        sharing_alpha=1.0,
        max_per_structure=None,
        genotypic_distance_threshold=0.25,
        structural_dedup_umbral=0.08,
        amplitude_penalty=0.35,
        amplitude_support_low=0.10,
        amplitude_support_high=0.80,
        amplitude_width_power=2.0,
    ):
        self.infected = []
        self.recovered = []
        self.deaths = []
        self.max_time = max_time
        self.data = data
        self.size = (len(data.columns)) * 2
        self.n_solutions = n_solutions
        self.bestSolutions = []
        self.bestSolutionEachIteration = []
        self.meanEachIteration = []
        self.stddevEachIteration = []
        self.avgBestFitnessDistance = []
        self.objF = objF
        # Dynamic niching (structural diversity inside the evolutionary loop)
        self.niching = bool(niching)
        self.sharing_radius = float(sharing_radius)
        self.sharing_alpha = float(sharing_alpha)
        self.max_per_structure = (
            max(1, int(max_per_structure))
            if max_per_structure is not None
            else max(1, int(n_solutions) // 4)
        )
        self.genotypic_distance_threshold = float(genotypic_distance_threshold)
        self.structural_dedup_umbral = float(structural_dedup_umbral)
        # Penalize wide / catch-all active intervals (data assumed normalized to [0,1]).
        # Weight is always adaptive in rule support when amplitude_penalty > 0.
        self.amplitude_penalty = max(0.0, float(amplitude_penalty))
        self.amplitude_support_low = float(amplitude_support_low)
        self.amplitude_support_high = float(amplitude_support_high)
        self.amplitude_width_power = max(1e-6, float(amplitude_width_power))
        if self.amplitude_support_high < self.amplitude_support_low:
            self.amplitude_support_low, self.amplitude_support_high = (
                self.amplitude_support_high,
                self.amplitude_support_low,
            )

    # --- fitness helpers / elite archive ---

    def _raw_fitness_of(self, individual):
        raw = getattr(individual, "raw_fitness", None)
        if raw is not None:
            return raw
        return self.fitness(individual.values, individual.attributeType)

    def _store_copy(self, candidate, cand_raw):
        stored = deepcopy(candidate)
        stored.raw_fitness = cand_raw
        stored.fitness = cand_raw
        return stored

    def es_regla_distinta(self, nuevo_individuo, umbral_distancia=None):
        """
        Comprueba si el nuevo individuo es lo suficientemente distinto
        de las reglas que ya estan en el top (bestSolutions).
        """
        if umbral_distancia is None:
            umbral_distancia = self.structural_dedup_umbral
        if not self.bestSolutions:
            return True

        for regla_top in self.bestSolutions:
            misma_estructura = np.array_equal(
                nuevo_individuo.attributeType, regla_top.attributeType
            )
            distancia = self.calcular_distancia(nuevo_individuo.values, regla_top.values)
            if misma_estructura and distancia < umbral_distancia:
                return False
        return True

    def try_insert_best_solution(self, candidate) -> bool:
        """
        Insert/replace in bestSolutions using raw fitness + structural niche capacity.
        Returns True if the archive changed.
        """
        if candidate is None:
            return False
        if any(candidate is x or candidate == x for x in self.bestSolutions):
            return False
        if not self.es_regla_distinta(candidate):
            return False

        cand_raw = self._raw_fitness_of(candidate)
        if cand_raw is None or (isinstance(cand_raw, float) and np.isnan(cand_raw)):
            return False

        fp = structure_fingerprint(candidate.attributeType)
        n_same = count_structure_in_archive(self.bestSolutions, fp)

        if n_same < self.max_per_structure:
            if len(self.bestSolutions) < self.n_solutions:
                self.bestSolutions.append(self._store_copy(candidate, cand_raw))
            else:
                peor_idx = min(
                    range(len(self.bestSolutions)),
                    key=lambda i: self._raw_fitness_of(self.bestSolutions[i]),
                )
                peor_fit = self._raw_fitness_of(self.bestSolutions[peor_idx])
                if cand_raw > peor_fit:
                    self.bestSolutions[peor_idx] = self._store_copy(candidate, cand_raw)
                else:
                    return False
        else:
            worst_idx, worst_ind = worst_in_structure(self.bestSolutions, fp)
            if worst_idx is None:
                return False
            worst_fit = self._raw_fitness_of(worst_ind)
            if cand_raw > worst_fit:
                self.bestSolutions[worst_idx] = self._store_copy(candidate, cand_raw)
            else:
                return False

        self.bestSolutions = sorted(
            self.bestSolutions,
            key=lambda ind: self._raw_fitness_of(ind),
            reverse=True,
        )
        return True

    # --- disease propagation ---

    def _pick_infection_donor(self, ranked_infected, current_idx):
        """
        Restricted donor selection adapted to CVOA:
        prefer a donor whose structure differs enough from the current individual.
        Falls back to the current individual (local interval search).
        """
        current = ranked_infected[current_idx]
        if not self.niching or self.genotypic_distance_threshold <= 0:
            return current

        window = ranked_infected[: max(3, min(15, len(ranked_infected)))]
        random.shuffle(window)
        for cand in window:
            if cand is current:
                continue
            if structural_distance(
                current.attributeType, cand.attributeType
            ) >= self.genotypic_distance_threshold:
                return cand
        return current

    def _is_new_candidate(self, ind, new_infected_list):
        return (
            ind not in self.deaths
            and ind not in self.infected
            and ind not in new_infected_list
            and ind not in self.recovered
        )

    def _try_add_infected(self, ind, new_infected_list):
        """Add a fresh infection or reinfect from recovered (same RNG order as before)."""
        if self._is_new_candidate(ind, new_infected_list):
            new_infected_list.append(ind)
            return
        if ind in self.recovered and ind not in new_infected_list:
            if random.random() < self.P_REINFECTION:
                new_infected_list.append(ind)
                self.recovered.remove(ind)

    def _try_isolate(self, ind, new_infected_list):
        if self._is_new_candidate(ind, new_infected_list):
            self.recovered.append(ind)

    def _spread_from_individual(self, x, i, idx_super_spreader, time, new_infected_list):
        if i < idx_super_spreader:
            ninfected = self.MIN_SUPERSPREAD + random.randint(
                0, self.MAX_SUPERSPREAD - self.MIN_SUPERSPREAD
            )
        else:
            ninfected = random.randint(0, self.MAX_SPREAD)

        traveler = random.random() < self.P_TRAVEL
        if traveler:
            travel_distance = random.randint(1, int(self.size / 2))
        else:
            travel_distance = 1

        donor = self._pick_infection_donor(self.infected, i)
        for _j in range(ninfected):
            new_infected = donor.infect(travel_distance=travel_distance)
            if time < self.SOCIAL_DISTANCING:
                self._try_add_infected(new_infected, new_infected_list)
            else:
                if random.random() > self.P_ISOLATION:
                    self._try_add_infected(new_infected, new_infected_list)
                else:
                    self._try_isolate(new_infected, new_infected_list)

    def propagateDisease(self, time):
        new_infected_list = []
        # Step 1. Assess fitness for each individual.
        for x in list(self.infected):
            x.fitness = self.fitness(x.values, x.attributeType)
            x.raw_fitness = x.fitness
            if np.isnan(x.fitness):
                self.deaths.append(x)
                self.infected.remove(x)

        if not self.infected:
            return

        # Step 1.1 Dynamic Fitness Sharing on structural niches
        if self.niching and len(self.infected) > 1:
            apply_fitness_sharing(
                self.infected,
                sharing_radius=self.sharing_radius,
                alpha=self.sharing_alpha,
            )

        ranked = sorted(self.infected, key=lambda i: i.fitness, reverse=True)
        self.infected = ranked
        self.bestSolutionEachIteration.append(self._raw_fitness_of(self.infected[0]))
        total_fitness = sum(self._raw_fitness_of(i) for i in self.infected)
        mean_fitness = total_fitness / len(self.infected)
        self.meanEachIteration.append(mean_fitness)
        std_dev_fitness = np.std([self._raw_fitness_of(i) for i in self.infected])
        self.stddevEachIteration.append(std_dev_fitness)

        # Step 2/3 Update elite archive with structural niche capacity
        for ind in self.infected:
            self.try_insert_best_solution(ind)

        if not self.bestSolutions:
            self.try_insert_best_solution(self.infected[0])

        self.avgBestFitnessDistance.append(self.avgBestFitnessDist())

        if len(self.infected) == 1:
            idx_super_spreader = 1
        else:
            idx_super_spreader = self.SUPERSPREADER_PERC * len(self.infected)
        if len(self.infected) == 1:
            idx_deaths = sys.maxsize
        else:
            idx_deaths = len(self.infected) - (self.DEATH_PERC * len(self.infected))

        # Step 5. Disease propagation (ordered by shared fitness when niching is on).
        still_infected = []
        for i, x in enumerate(list(self.infected)):
            if i >= idx_deaths:
                self.deaths.append(x)
            else:
                still_infected.append(x)
                self._spread_from_individual(
                    x, i, idx_super_spreader, time, new_infected_list
                )

        self.recovered.extend(still_infected)
        self.infected = new_infected_list

    # --- run loop ---

    def _print_variant_banner(self):
        if self.niching:
            print(
                "Niching ON: fitness_sharing "
                f"(radius={self.sharing_radius}, alpha={self.sharing_alpha}), "
                f"max_per_structure={self.max_per_structure}, "
                f"genotypic_distance_threshold={self.genotypic_distance_threshold}"
            )
        else:
            print("Niching OFF")
        if self.amplitude_penalty > 0:
            print(
                f"Amplitude penalty ON (adaptive): weight_max={self.amplitude_penalty}, "
                f"support_low={self.amplitude_support_low}, "
                f"support_high={self.amplitude_support_high}, "
                f"width_power={self.amplitude_width_power} "
                "(fitness *= 1 - W_eff(support) * mean_width**power)"
            )
        else:
            print("Amplitude penalty OFF")

    def _init_patient_zero(self):
        pz = Individual.random(self.data)
        while (
            Individual.validateAttributeTypes(pz, pz.attributeType) == 0
            or self.fitness(pz.values, pz.attributeType) == 0
        ):
            pz = Individual.random(self.data)
        pz.fitness = self.fitness(pz.values, pz.attributeType)
        pz.raw_fitness = pz.fitness
        self.infected.append(pz)
        print("Patient Zero: " + str(pz) + "\n")
        print("Patient Zero attribute values: " + str(pz.values) + "\n")
        print("Patient Zero attribute type: " + str(pz.attributeType) + "\n")
        self.try_insert_best_solution(pz)
        return pz

    def _print_iteration_status(self, time, current_best_fitness):
        print("Iteration ", (time + 1))
        print("Best fitness so far: ", current_best_fitness)
        print("Best individual: ", self.bestSolutions[0].kintegers)
        if self.niching:
            n_structs = len(
                {structure_fingerprint(x.attributeType) for x in self.bestSolutions}
            )
            print(
                f"Elite structures: {n_structs}/{len(self.bestSolutions)} "
                f"(max_per_structure={self.max_per_structure})"
            )
        print(
            "Infected: ",
            str(len(self.infected)),
            "; Recovered: ",
            str(len(self.recovered)),
            "; Deaths: ",
            str(len(self.deaths)),
        )
        print(
            "Recovered/Infected: "
            + str(
                "{:.4f}".format(
                    100 * ((len(self.recovered)) / (len(self.infected) + 0.01))
                )
                + "%"
            )
        )

    def _update_early_stop(self, best_fitness_history, consecutive_stable_iterations, epsilon, patience):
        """Return (epidemic_continues, consecutive_stable_iterations)."""
        if len(best_fitness_history) <= 1:
            return True, consecutive_stable_iterations
        improvement = best_fitness_history[-1] - best_fitness_history[-2]
        if improvement < epsilon:
            consecutive_stable_iterations += 1
        else:
            consecutive_stable_iterations = 0
        if consecutive_stable_iterations >= patience:
            print("Fitness se ha estabilizado. Parando el proceso.")
            return False, consecutive_stable_iterations
        return True, consecutive_stable_iterations

    def run(self):
        epidemic = True
        time = 0

        patience = max(3, int(self.max_time * 0.20))
        consecutive_stable_iterations = 0
        epsilon = 1e-6
        best_fitness_history = []

        self._print_variant_banner()
        self._init_patient_zero()

        while epidemic and time < self.max_time:
            self.propagateDisease(time)

            if not self.bestSolutions:
                epidemic = False
                break

            current_best_fitness = self._raw_fitness_of(self.bestSolutions[0])
            best_fitness_history.append(current_best_fitness)
            self._print_iteration_status(time, current_best_fitness)

            epidemic, consecutive_stable_iterations = self._update_early_stop(
                best_fitness_history,
                consecutive_stable_iterations,
                epsilon,
                patience,
            )

            if not self.infected:
                epidemic = False
            time += 1
        return self.bestSolutions

    def getBestFitnessEachIt(self):
        return self.bestSolutionEachIteration

    def getMeanFitnessEachIt(self):
        return self.meanEachIteration

    def getStdFitnessEachIt(self):
        return self.stddevEachIteration

    # --- fitness / objectives ---

    def mean_active_interval_width(self, individual_values, individual_attributeType):
        """Mean width of antecedent/consequent intervals (normalized [0,1] space)."""
        widths = []
        n = len(individual_attributeType) // 2
        for i in range(n):
            role = individual_attributeType[i * 2]
            if role not in (1, 2):
                continue
            lo = float(individual_values[i * 2])
            hi = float(individual_values[i * 2 + 1])
            widths.append(max(0.0, min(1.0, hi - lo)))
        if not widths:
            return 1.0
        return sum(widths) / float(len(widths))

    def effective_amplitude_weight(self, rule_support_frac):
        """Adaptive W(support): 0 if low support, W_max if high support, linear in between."""
        w_max = self.amplitude_penalty
        if w_max <= 0:
            return 0.0
        s = max(0.0, min(1.0, float(rule_support_frac)))
        lo = self.amplitude_support_low
        hi = self.amplitude_support_high
        if s <= lo:
            return 0.0
        if s >= hi or hi <= lo:
            return w_max
        return w_max * ((s - lo) / (hi - lo))

    def amplitude_multiplier(self, individual_values, individual_attributeType, rule_support_frac=0.0):
        """Return factor in [1-W_eff, 1]: narrower active intervals => closer to 1."""
        w = self.effective_amplitude_weight(rule_support_frac)
        if w <= 0:
            return 1.0
        mean_w = self.mean_active_interval_width(individual_values, individual_attributeType)
        return 1.0 - w * (mean_w ** self.amplitude_width_power)

    def fitness(self, individual_values, individual_attributeType):
        X = self.data.to_numpy(dtype=float, copy=False)
        n_rows, n_cols = X.shape

        low = np.asarray(individual_values[0::2], dtype=float)
        high = np.asarray(individual_values[1::2], dtype=float)
        types = np.asarray(individual_attributeType[0::2])

        within = (X >= low) & (X <= high)

        ant_cols = types == 1
        cons_cols = types == 2

        if ant_cols.any():
            mask_ant = within[:, ant_cols].all(axis=1)
        else:
            mask_ant = np.ones(n_rows, dtype=bool)

        if cons_cols.any():
            mask_cons = within[:, cons_cols].all(axis=1)
        else:
            mask_cons = np.ones(n_rows, dtype=bool)

        support_ant = int(mask_ant.sum())
        support_cons = int(mask_cons.sum())
        support_rule = int((mask_ant & mask_cons).sum())

        conf = (support_rule / support_ant) if support_ant != 0 else 0.0

        if self.objF == "1":
            base = self.objectiveFunc1(support_ant, support_cons, support_rule, conf)
        elif self.objF == "3":
            base = self.objectiveFunc3(support_ant, support_cons, support_rule, conf)
        else:
            # objF "2" (and any legacy default)
            base = self.objectiveFunc2(support_ant, support_cons, support_rule, conf)

        if base == 0 or self.amplitude_penalty <= 0:
            return base

        rule_sup_frac = (support_rule / n_rows) if n_rows else 0.0
        mult = self.amplitude_multiplier(
            individual_values, individual_attributeType, rule_support_frac=rule_sup_frac
        )
        # Narrower intervals must always score better (higher), including when base < 0.
        if base > 0:
            return base * mult
        return base / max(mult, 1e-12)

    def objectiveFunc1(self, support_ant, support_cons, support_rule, conf):
        leverage = ((support_rule * len(self.data.index)) - (support_ant * support_cons)) / pow(
            len(self.data.index), 2
        )
        accuracy = (
            support_rule + (len(self.data.index) - (support_ant + support_cons - support_rule))
        ) / len(self.data.index)
        return accuracy + conf + leverage

    def objectiveFunc2(self, support_ant, support_cons, support_rule, conf):
        if support_ant != 0:
            if conf > support_cons / len(self.data.index):
                cf = ((support_rule * len(self.data.index)) - (support_ant * support_cons)) / (
                    (len(self.data.index) - support_cons) * support_ant
                )
            else:
                if support_cons != 0:
                    cf = ((support_rule * len(self.data.index)) - (support_ant * support_cons)) / (
                        support_ant * support_cons
                    )
                else:
                    cf = 0
        else:
            cf = 0
        support = support_rule / len(self.data.index)
        return cf + conf + support

    def objectiveFunc3(self, support_ant, support_cons, support_rule, conf):
        """fobj3: support + conf + netconf (VLMOHSNAR-oriented interest + coverage)."""
        n = len(self.data.index)
        if n == 0:
            return 0.0
        support = support_rule / n
        p_ant = support_ant / n
        p_cons = support_cons / n
        leverage = support - (p_ant * p_cons)
        den_net = p_ant * (1.0 - p_ant)
        netconf = (leverage / den_net) if den_net > 1e-12 else 0.0
        return support + conf + netconf

    # --- distances / getters ---

    def avgBestFitnessDist(self):
        distancias = []
        if len(self.bestSolutions) > 1:
            for i in range(len(self.bestSolutions)):
                for j in range(i + 1, len(self.bestSolutions)):
                    distancia = self.calcular_distancia(
                        self.bestSolutions[i].values, self.bestSolutions[j].values
                    )
                    distancias.append(distancia)
            promedio_distancia = sum(distancias) / len(distancias)
        else:
            promedio_distancia = 0
        return promedio_distancia

    def calcular_distancia(self, regla1, regla2):
        regla1 = np.array(regla1)
        regla2 = np.array(regla2)
        return np.linalg.norm(regla1 - regla2)

    def getAvgBestFitnessDist(self):
        return self.avgBestFitnessDistance
