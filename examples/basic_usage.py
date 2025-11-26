"""
Basic usage example for TIM matching
"""

import pandas as pd
import numpy as np
from tim import TIMatcher

# Generate sample data
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'age': np.random.normal(45, 15, n),
    'income': np.random.normal(50000, 20000, n),
    'bmi': np.random.normal(25, 5, n),
    'gender': np.random.choice([0, 1], n),
    'education': np.random.choice([0, 1, 2], n),
})

# Generate treatment (influenced by covariates)
treatment_prob = 1 / (1 + np.exp(-(
    0.02 * data['age'] + 
    0.00001 * data['income'] + 
    0.1 * data['bmi'] +
    0.5 * data['gender'] +
    0.3 * data['education'] - 3
)))
data['treatment'] = np.random.binomial(1, treatment_prob)

# Generate outcome (influenced by treatment and covariates)
data['outcome'] = (
    2.0 * data['treatment'] +  # Treatment effect
    0.05 * data['age'] +
    0.00005 * data['income'] +
    0.2 * data['bmi'] +
    1.0 * data['gender'] +
    0.5 * data['education'] +
    np.random.normal(0, 2, n)
)

# Initialize and fit matcher
matcher = TIMatcher(
    treatment_col='treatment',
    outcome_col='outcome',
    continuous_cols=['age', 'income', 'bmi'],
    discrete_cols=['gender', 'education'],
    coarsen_bins=4
)

matcher.fit(data)
matcher.summary()

# Get matched data
matched_data = matcher.get_matched_data()
print(f"\nMatched data shape: {matched_data.shape}")
print(f"Treatment effect (ATE): {matcher.ate_:.4f}")
print(f"True treatment effect: 2.0000")
