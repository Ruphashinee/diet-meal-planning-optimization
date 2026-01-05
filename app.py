import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# --- DATA PREPROCESSING ---
@st.cache_data
def prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    # Define how we split daily nutrition into 4 meals
    splits = {
        'Breakfast Suggestion': 0.25, 
        'Lunch Suggestion': 0.35, 
        'Dinner Suggestion': 0.30, 
        'Snack Suggestion': 0.10
    }
    nutrients = ['Calories', 'Protein', 'Carbohydrates', 'Fat']
    
    food_db = {}
    diet_map = {}
    
    for meal_type, split in splits.items():
        # Calculate nutritional averages for each unique meal name
        meal_stats = df.groupby(meal_type)[nutrients].mean() * split
        food_db[meal_type] = meal_stats.reset_index()
        
        # Map meals to allowed dietary preferences
        mapping = df.groupby(meal_type)['Dietary Preference'].apply(lambda x: set(x)).to_dict()
        diet_map[meal_type] = mapping
        
    return food_db, diet_map

# --- EVOLUTIONARY PROGRAMMING ALGORITHM ---
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
            
            # Check dietary preference (Extended Analysis: Constraints)
            allowed_diets = diet_map[mt].get(meal_name, set())
            if user_diet not in allowed_diets and user_diet != "Omnivore":
                penalty += 10000 
            
            for n in target_nutrients.keys():
                total_nutrients[n] += meal_row[n]
        
        # Multi-Objective Fitness (Weighted Error)
        error = 0
        for n, target in target_nutrients.items():
            error += weights.get(n, 1.0) * abs(target - total_nutrients[n])
        return error + penalty

    # Step 1: Initialize Population
    population = [[random.randint(0, db_sizes[mt]-1) for mt in meal_types] for _ in range(pop_size)]
    fitness_history = []
    
    start_time = time.time()
    for gen in range(generations):
        fits = [get_fitness(ind) for ind in population]
        best_fit = min(fits)
        fitness_history.append(best_fit)
        
        # Step 2: EP Mutation (Create Offspring)
        offspring = []
        for ind in population:
            child = ind.copy()
            for _ in range(random.randint(1, 2)):
                pos = random.randint(0, len(meal_types)-1)
                child[pos] = random.randint(0, db_sizes[meal_types[pos]]-1)
            offspring.append(child)
        
        # Step 3: (mu + lambda) Selection
        combined_pop = population + offspring
        combined_fits = fits + [get_fitness(ind) for ind in offspring]
        sorted_indices = np.argsort(combined_fits)
        population = [combined_pop[i] for i in sorted_indices[:pop_size]]
    
    end_time = time.time()
    return population[0], fitness_history, end_time - start_time

# --- STREAMLIT UI ---
st.set_page_config(page_title="Diet Optimizer", layout="wide")
st.title("🥗 Diet Meal Planning Optimization (EP)")

try:
    food_db, diet_map = prepare_data('Food_and_Nutrition__.csv')
    
    with st.sidebar:
        st.header("User Profile")
        diet = st.selectbox("Dietary Preference", ["Omnivore", "Vegetarian", "Vegan"])
        target_cals = st.number_input("Target Daily Calories", 1200, 4000, 2000)
        
        st.header("EP Algorithm Parameters")
        pop_size = st.slider("Population Size", 10, 100, 50)
        gens = st.slider("Generations", 50, 500, 100)

    if st.button("🚀 Optimize My Meal Plan"):
        # Targets based on user input
        targets = {
            'Calories': target_cals, 
            'Protein': (target_cals * 0.25) / 4, 
            'Carbohydrates': (target_cals * 0.50) / 4, 
            'Fat': (target_cals * 0.25) / 9
        }
        weights = {'Calories': 1.0, 'Protein': 2.0, 'Carbohydrates': 1.0, 'Fat': 1.0}
        
        best_ind, history, duration = run_ep(targets, diet, food_db, diet_map, pop_size, gens, weights)
        
        st.success(f"Optimized in {duration:.2f} seconds")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Recommended Plan")
            for i, mt in enumerate(food_db.keys()):
                st.info(f"**{mt}:** {food_db[mt].iloc[best_ind[i]][mt]}")
        
        with col2:
            st.subheader("Convergence Analysis")
            fig, ax = plt.subplots()
            ax.plot(history, color='green')
            ax.set_title("Fitness Over Generations (Minimizing Error)")
            st.pyplot(fig)

except FileNotFoundError:
    st.error("Please upload 'Food_and_Nutrition__.csv' to the same folder as this script.")
