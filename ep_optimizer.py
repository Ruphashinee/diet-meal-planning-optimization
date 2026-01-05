import pandas as pd
import numpy as np

df = pd.read_csv("Food_and_Nutrition__.csv")

def fitness(solution, cal_target, protein_min, budget):
    calories = np.sum(solution * df["Calories"])
    protein = np.sum(solution * df["Protein"])
    cost = np.sum(solution * df["Cost"])

    penalty = abs(calories - cal_target)

    if protein < protein_min:
        penalty += (protein_min - protein) * 10

    if cost > budget:
        penalty += (cost - budget) * 20

    return penalty

def run_ep(cal_target, protein_min, budget,
           pop_size=30, generations=50):

    n_foods = len(df)
    population = np.random.rand(pop_size, n_foods)
    best_history = []

    for _ in range(generations):
        scores = [fitness(ind, cal_target, protein_min, budget)
                  for ind in population]

        best_history.append(min(scores))

        parents = population[np.argsort(scores)[:10]]
        offspring = parents + np.random.normal(0, 0.1, parents.shape)
        population = np.clip(offspring, 0, 1)

    best_idx = np.argmin(
        [fitness(ind, cal_target, protein_min, budget)
         for ind in population]
    )

    return population[best_idx], best_history
