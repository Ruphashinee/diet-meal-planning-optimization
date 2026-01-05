# Diet Meal Planning Optimisation using EP

## Case Study
This project optimizes daily meal plans based on nutritional targets using **Evolutionary Programming (EP)**. It uses a dataset of food suggestions to minimize the difference between target macros and actual meal values.

## Algorithm Details
- **Method:** Evolutionary Programming (EP)
- **Selection:** (μ + λ) Selection
- **Mutation:** Random search-based index swapping
- **Fitness:** Weighted Absolute Error across 4 nutritional objectives.

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run app: `streamlit run app.py`
