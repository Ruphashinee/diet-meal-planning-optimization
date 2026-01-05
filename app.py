import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import time

# Load data
@st.cache_data
def load_data():
    with open('food_data.pkl', 'rb') as f:
        return pickle.load(f)

data = load_data()
food_db = data['food_db']
diet_map = data['diet_map']

def run_ep(target_nutrients, user_diet, food_db, diet_map, pop_size, generations, weights):
    meal_types = list(food_db.keys())
    db_sizes = {mt: len(food_db[mt]) for mt in meal_types}

    def get_fitness(individual):
        total_nutrients = {n: 0 for n in target_nutrients.keys()}
        penalty = 0
        for i, mt in enumerate(meal_types):
            meal_idx = individual[i]
            meal_row = food_db[mt].iloc[meal_idx]
            meal_name = meal_row[mt]
            allowed_diets = diet_map[mt].get(meal_name, set())
            if user_diet not in allowed_diets and user_diet != "Omnivore":
                penalty += 5000 
            for n in target_nutrients.keys():
                total_nutrients[n] += meal_row[n]
        
        error = 0
        for n, target in target_nutrients.items():
            error += weights.get(n, 1.0) * abs(target - total_nutrients[n])
        return error + penalty

    # EP Initialization
    population = [[random.randint(0, db_sizes[mt]-1) for mt in meal_types] for _ in range(pop_size)]
    fitness_history = []
    
    start_time = time.time()
    for gen in range(generations):
        fits = [get_fitness(ind) for ind in population]
        best_fit = min(fits)
        fitness_history.append(best_fit)
        
        # EP Mutation (Primary Operator)
        offspring = []
        for ind in population:
            child = ind.copy()
            for _ in range(random.randint(1, 2)):
                pos = random.randint(0, len(meal_types)-1)
                child[pos] = random.randint(0, db_sizes[meal_types[pos]]-1)
            offspring.append(child)
        
        # (mu + lambda) Selection
        combined_pop = population + offspring
        combined_fits = fits + [get_fitness(ind) for ind in offspring]
        sorted_indices = np.argsort(combined_fits)
        population = [combined_pop[i] for i in sorted_indices[:pop_size]]
    
    end_time = time.time()
    return population[0], fitness_history, end_time - start_time

# Streamlit UI
st.title("🥗 Diet Optimizer (Evolutionary Programming)")

with st.sidebar:
    st.header("Settings")
    diet = st.selectbox("Diet", ["Omnivore", "Vegetarian", "Vegan"])
    target_cals = st.number_input("Target Calories", 1200, 4000, 2000)
    pop_size = st.slider("Population", 10, 100, 50)
    gens = st.slider("Generations", 50, 500, 100)

if st.button("Generate Meal Plan"):
    targets = {'Calories': target_cals, 'Protein': target_cals*0.03, 'Carbohydrates': target_cals*0.12, 'Fat': target_cals*0.03}
    weights = {'Calories': 1.0, 'Protein': 2.0, 'Carbohydrates': 1.0, 'Fat': 1.0}
    
    best_ind, history, duration = run_ep(targets, diet, food_db, diet_map, pop_size, gens, weights)
    
    st.success(f"Optimized in {duration:.2f}s")
    st.line_chart(history)
    
    for i, mt in enumerate(food_db.keys()):
        st.write(f"**{mt}:** {food_db[mt].iloc[best_ind[i]][mt]}")
