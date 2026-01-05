import pandas as pd
import numpy as np
import random

# Load your specific CSV
df = pd.read_csv("Food_and_Nutrition__.csv")

def run_diet_ep(target_calories, user_diet, pop_size=20, generations=100):
    # 1. Initialization: Create random meal plans (indices of rows in CSV)
    # Each individual is [breakfast_idx, lunch_idx, dinner_idx, snack_idx]
    n_rows = len(df)
    population = [[random.randint(0, n_rows-1) for _ in range(4)] for _ in range(pop_size)]
    
    best_score_history = []

    for gen in range(generations):
        # 2. Fitness Evaluation
        fitness_scores = []
        for individual in population:
            total_cal = 0
            diet_penalty = 0
            
            for idx in individual:
                row = df.iloc[idx]
                total_cal += row['Calories']
                
                # Constraint: Check if food matches user diet (e.g., Vegan)
                if user_diet != "Omnivore" and row['Dietary Preference'] != user_diet:
                    diet_penalty += 5000 # High penalty for wrong food type
            
            # Error = Difference from target + Diet penalties
            fitness = abs(total_cal - target_calories) + diet_penalty
            fitness_scores.append(fitness)

        # 3. Selection (Keep the best parents)
        best_indices = np.argsort(fitness_scores)[:pop_size//2]
        parents = [population[i] for i in best_indices]
        best_score_history.append(min(fitness_scores))

        # 4. Mutation (EP core: Create offspring by slightly changing parents)
        offspring = []
        for p in parents:
            child = list(p)
            # Mutate: Pick one meal in the day and change it to a random new one
            mutate_slot = random.randint(0, 3)
            child[mutate_slot] = random.randint(0, n_rows-1)
            offspring.append(child)

        # New population = Parents + Offspring
        population = parents + offspring

    # Return the best solution found
    final_scores = [fitness(ind, target_calories, user_diet) for ind in population] # (concept)
    best_plan = population[np.argmin(fitness_scores)]
    
    return df.iloc[best_plan], best_score_history
