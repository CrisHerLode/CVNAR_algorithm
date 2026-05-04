from copy import deepcopy
import numpy as np
from numpy import mean
import sys as sys
import random as random
from individual import Individual
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

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

    def __init__(self, max_time, data, n_solutions, objF):
        self.infected = []
        self.recovered = []
        self.deaths = []
        self.max_time = max_time
        self.data = data
        self.size = (len(data.columns))*2
        self.n_solutions = n_solutions
        self.bestSolutions = []
        self.bestSolutionEachIteration = []
        self.meanEachIteration = []
        self.stddevEachIteration = []
        self.avgBestFitnessDistance = []
        self.objF = objF
    
    def es_regla_distinta(self, nuevo_individuo, umbral_distancia=0.08):
        """
        Comprueba si el nuevo individuo es lo suficientemente distinto
        de las reglas que ya están en el top (bestSolutions).
        """
        if not self.bestSolutions:
            return True
            
        for regla_top in self.bestSolutions:
            # Comparamos la estructura (si usan los mismos atributos como antecedente/consecuente)
            misma_estructura = np.array_equal(nuevo_individuo.attributeType, regla_top.attributeType)
            
            # Comparamos la distancia de los intervalos
            distancia = self.calcular_distancia(nuevo_individuo.values, regla_top.values)
            
            # Si tienen la misma estructura y sus valores son casi idénticos (clones)
            if misma_estructura and distancia < umbral_distancia:
                return False # No es distinta, la rechazamos
                
        return True # Ha superado el filtro, es una regla novedosa
    
    def propagateDisease(self, time):
        new_infected_list = []
        # Step 1. Assess fitness for each individual.
        for x in self.infected:
            x.fitness = self.fitness(x.values,x.attributeType)
            # If x.fitness is NaN, move from infected list to deaths lists
            if np.isnan(x.fitness):
                self.deaths.append(x)
                self.infected.remove(x)
        
        # Step 2. Sort the infected list by fitness (descendent).
        self.infected = sorted(self.infected, key=lambda i: i.fitness, reverse=True)
        self.bestSolutionEachIteration.append(self.infected[0].fitness)
        total_fitness = sum(i.fitness for i in self.infected)
        mean_fitness = total_fitness / len(self.infected)
        self.meanEachIteration.append(mean_fitness)
        std_dev_fitness = np.std([i.fitness for i in self.infected])
        self.stddevEachIteration.append(std_dev_fitness)
        
        # Step 2.1 Add individuals to the bestSolutions until n_solutions is reached
        i=0
        while (len(self.bestSolutions)<self.n_solutions) and i<(len(self.infected)):
            if self.infected[i] not in self.bestSolutions:
                if self.es_regla_distinta(self.infected[i]):
                    self.bestSolutions.append(deepcopy(self.infected[i]))
            i+=1
            
        self.bestSolutions = sorted(self.bestSolutions, key=lambda i: self.fitness(i.values,i.attributeType), reverse=True)
        # Step 3. Update best global solutions, if proceed.
        if self.n_solutions > 1:
            # Iteramos sobre los infectados para ver si alguno merece entrar al top
            for i in range(len(self.infected)):
                # Comprobamos si es mejor que el PEOR de nuestra lista top (que es el último [-1])
                peor_top = self.bestSolutions[-1]
                peor_fitness = self.fitness(peor_top.values, peor_top.attributeType)
                if self.infected[i].fitness > peor_fitness and self.infected[i] not in self.bestSolutions:
                    if self.es_regla_distinta(self.infected[i]):
                        # Reemplazamos a la peor regla y volvemos a ordenar
                        self.bestSolutions[-1] = deepcopy(self.infected[i])
                        self.bestSolutions = sorted(self.bestSolutions, key=lambda ind: self.fitness(ind.values, ind.attributeType), reverse=True)
        else:
            best0_fitness = self.fitness(self.bestSolutions[0].values, self.bestSolutions[0].attributeType)
            if best0_fitness is None or self.infected[0].fitness > best0_fitness:
                self.bestSolutions[0] = deepcopy(self.infected[0])
        # Step 3.1 Calculate distance between the best solutions
        self.avgBestFitnessDistance.append(self.avgBestFitnessDist())
        # Step 4. Assess indexes to point super-spreaders and deaths parts of the infected list.
        if len(self.infected)==1:
            idx_super_spreader=1
        else:
            idx_super_spreader = self.SUPERSPREADER_PERC * len(self.infected)
        if len(self.infected) == 1:
            idx_deaths = sys.maxsize
        else:
            idx_deaths = len(self.infected) - (self.DEATH_PERC * len(self.infected))
        
        # Step 5. Disease propagation.
        i = 0
        for x in self.infected:
            # Step 5.1 If the individual belongs to the death part, then die!
            if i >= idx_deaths:
                self.deaths.append(x)
                self.infected.remove(x)
            else:
                # Step 5.2 Determine the number of new infected individuals.
                if i < idx_super_spreader:  # This is the super-spreader!
                    ninfected = self.MIN_SUPERSPREAD + random.randint(0, self.MAX_SUPERSPREAD - self.MIN_SUPERSPREAD)
                else:
                    ninfected = random.randint(0, self.MAX_SPREAD)
                # Step 5.3 Determine whether the individual has traveled
                if random.random() < self.P_TRAVEL:
                    traveler = True
                else:
                    traveler = False
                # Step 5.4 Determine the travel distance, which indicates how many intervals of an individual will be infected.
                if traveler:
                    travel_distance = random.randint(1,self.size/2) 
                else:
                    travel_distance = 1 #The individual has not travel
                # Step 5.5 Infect!!
                for j in range(ninfected):
                    new_infected = x.infect(travel_distance=travel_distance)  # new_infected = infect(x, travel_distance)
                    # Propagate with no social distancing measures
                    if time < self.SOCIAL_DISTANCING:
                        if new_infected not in self.deaths and new_infected not in self.infected and new_infected not in new_infected_list and new_infected not in self.recovered:
                            new_infected_list.append(new_infected)
                        elif new_infected in self.recovered and new_infected not in new_infected_list:
                            if random.random() < self.P_REINFECTION:
                                new_infected_list.append(new_infected)
                                self.recovered.remove(new_infected)
                    else: # After SOCIAL_DISTANCING iterations, there is a P_ISOLATION of not being infected
                        if random.random() > self.P_ISOLATION:
                            if new_infected not in self.deaths and new_infected not in self.infected and new_infected not in new_infected_list and new_infected not in self.recovered:
                                new_infected_list.append(new_infected)
                            elif new_infected in self.recovered and new_infected not in new_infected_list:
                                if random.random() < self.P_REINFECTION:
                                    new_infected_list.append(new_infected)
                                    self.recovered.remove(new_infected)
                        else: # Those saved by social distancing are sent to the recovered list
                            if new_infected not in self.deaths and new_infected not in self.infected and new_infected not in new_infected_list and new_infected not in self.recovered:
                                self.recovered.append(new_infected)
            i+=1
            
        # Step 6. Add the current infected individuals to the recovered list.
        self.recovered.extend(self.infected)
        # Step 7. Update the infected list with the new infected individuals.
        self.infected = new_infected_list
    
    def run(self):
        epidemic = True
        time = 0

        # Inicializar variables configuración early stopping
        patience = max(3, int(self.max_time * 0.20))
        consecutive_stable_iterations = 0
        epsilon = 1e-6
        best_fitness_history = []
        

        # Step 1. Infect to Patient Zero
        pz = Individual.random(self.data)
        while Individual.validateAttributeTypes(pz,pz.attributeType) == 0 or self.fitness(pz.values, pz.attributeType) == 0:
            pz = Individual.random(self.data)
        pz.fitness = self.fitness(pz.values, pz.attributeType)
        self.infected.append(pz)
        print("Patient Zero: " + str(pz) + "\n")
        print("Patient Zero attribute values: " + str(pz.values) + "\n")
        print("Patient Zero attribute type: " + str(pz.attributeType) + "\n")
        self.bestSolutions.append(deepcopy(pz))
        # Step 2. The main loop for the disease propagation
        while epidemic and time < self.max_time:
            self.propagateDisease(time)

            # Recuperamos el mejor fitness actual
            current_best_fitness = self.fitness(self.bestSolutions[0].values,self.bestSolutions[0].attributeType)
            best_fitness_history.append(current_best_fitness)

            print("Iteration ", (time + 1))
            print("Best fitness so far: ",self.fitness(self.bestSolutions[0].values,self.bestSolutions[0].attributeType))
            print("Best individual: ", self.bestSolutions[0].kintegers)
            print("Infected: ", str(len(self.infected)), "; Recovered: ", str(len(self.recovered)), "; Deaths: ", str(len(self.deaths)))
            print("Recovered/Infected: " + str("{:.4f}".format(100 * ((len(self.recovered)) / (len(self.infected)+0.01))) + "%"))

            # Early stopping
            if len(best_fitness_history) > 1:
                # Comparamos la iteración actual con la iteración anterior
                improvement = best_fitness_history[-1] - best_fitness_history[-2]
                
                if improvement < epsilon:
                    consecutive_stable_iterations += 1
                else:
                    consecutive_stable_iterations = 0
                if consecutive_stable_iterations >= patience:
                    print("Fitness se ha estabilizado. Parando el proceso.")
                    epidemic = False

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
        
    def fitness(self, individual_values, individual_attributeType):
        support_ant = support_cons = support_rule = 0
        
        X = self.data.to_numpy(dtype=float, copy=False)  # (n_instancias, n_columnas)
        n_rows, n_cols = X.shape

        low = np.asarray(individual_values[0::2], dtype=float)   # (n_cols,)
        high = np.asarray(individual_values[1::2], dtype=float)  # (n_cols,)
        types = np.asarray(individual_attributeType[0::2])  

        # Celdas que cumplen el rango por columna
        within = (X >= low) & (X <= high)

        ant_cols = (types == 1)
        cons_cols = (types == 2)

        # Si no hay columnas de un tipo, la condición se considera True
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

        if self.objF == '1':
            return self.objectiveFunc1(support_ant, support_cons, support_rule, conf)
        else:
            return self.objectiveFunc2(support_ant, support_cons, support_rule, conf)
    
    def objectiveFunc1(self,support_ant,support_cons,support_rule,conf):
        leverage = ((support_rule*len(self.data.index)) - (support_ant*support_cons)) / pow(len(self.data.index),2)
        accuracy = (support_rule + (len(self.data.index)-(support_ant+support_cons-support_rule))) / len(self.data.index)
        metricResult = accuracy + conf + leverage
        return metricResult
    
    def objectiveFunc2(self,support_ant,support_cons,support_rule,conf):
        if support_ant !=0:
            if (conf > support_cons/len(self.data.index)):
                cf = ((support_rule*len(self.data.index)) - (support_ant*support_cons)) / ((len(self.data.index)-support_cons)*support_ant)
            else:
                if support_cons !=0:
                    cf = ((support_rule*len(self.data.index)) - (support_ant*support_cons)) / (support_ant*support_cons)
                else:
                    cf= 0 
        else:
            cf = 0
        support = support_rule / len(self.data.index)
        metricResult = cf + conf + support
        return metricResult
    
    def avgBestFitnessDist(self):
        distancias = []
        if len(self.bestSolutions) > 1:
            for i in range(len(self.bestSolutions)):
                for j in range(i + 1, len(self.bestSolutions)):
                    distancia = self.calcular_distancia(self.bestSolutions[i].values, self.bestSolutions[j].values)  # Usar Euclidiana o Hamming
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
                    
    
