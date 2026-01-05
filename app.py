import streamlit as st
from ep_optimizer import run_ep

st.title("Diet Meal Planning Optimization")

cal_target = st.slider("Target Calories", 1500, 3000, 2000)
protein_min = st.slider("Minimum Protein (g)", 50, 200, 100)
budget = st.slider("Budget (RM)", 10, 50, 30)

if st.button("Run Optimization"):
    best_solution, history = run_ep(cal_target, protein_min, budget)
    st.subheader("Optimization Progress")
    st.line_chart(history)
    st.subheader("Best Solution Vector")
    st.write(best_solution)
