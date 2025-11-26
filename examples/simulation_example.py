"""
Simulation example demonstrating TIM matching
"""

import numpy as np
import pandas as pd
from tim import TIMatcher

def generate_simulated_data(n=1000, seed=42):
    """Generate simulated data for testing TIM"""
    np.random.seed(seed)
    
    # Generate covariates
    data = pd.DataFrame({
        'age': np.random.normal(45, 15, n),
        'income': np.random.exponential(50000, n),
        'bmi': np.random.normal(25, 5, n),
        'blood_pressure': np.random.normal(120, 15, n),
        'cholesterol': np.random.normal(200, 40, n),
        'gender': np.random.choice([0, 1], n),
        'smoking': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'education': np.random.choice([0, 1, 2], n, p=[0.3, 0.5, 0.2]),
    })
    
    # Generate treatment (with confounding)
    treatment_logit = (
        -3 +
        0.02 * data['age'] + 
        0.00001 * data['income'] + 
        0.1 * data['bmi'] +
        0.05 * data['blood_pressure'] +
        0.01 * data['cholesterol'] +
        0.5 * data['gender'] +
        0.8 * data['smoking'] +
        0.3 * data['education']
    )
    treatment_prob = 1 / (1 + np.exp(-treatment_logit))
    data['treatment'] = np.random.binomial(1, treatment_prob)
    
    # Generate outcome (with treatment effect)
    true_ate = 2.5
    data['outcome'] = (
        true_ate * data['treatment'] +
        0.05 * data['age'] +
        0.00005 * data['income'] +
        0.2 * data['bmi'] +
        0.08 * data['blood_pressure'] +
        0.02 * data['cholesterol'] +
        1.0 * data['gender'] +
        1.5 * data['smoking'] +
        0.5 * data['education'] +
        np.random.normal(0, 2, n)
    )
    
    return data, true_ate

def run_simulation(n_iterations=10, n_samples=1000):
    """Run simulation study"""
    results = {
        'iteration': [],
        'estimated_ate': [],
        'estimated_att': [],
        'true_ate': [],
        'bias': [],
        'percent_bias': [],
        'overlap_initial': [],
        'overlap_final': [],
        'treatment_retention': []
    }
    
    for i in range(n_iterations):
        print(f"\nIteration {i+1}/{n_iterations}")
        
        # Generate data
        data, true_ate = generate_simulated_data(n=n_samples, seed=i)
        
        # Fit matcher
        matcher = TIMatcher(
            treatment_col='treatment',
            outcome_col='outcome',
            continuous_cols=['age', 'income', 'bmi', 'blood_pressure', 'cholesterol'],
            discrete_cols=['gender', 'smoking', 'education'],
            coarsen_bins=4
        )
        
        try:
            matcher.fit(data)
            
            # Store results
            results['iteration'].append(i+1)
            results['estimated_ate'].append(matcher.ate_)
            results['estimated_att'].append(matcher.att_)
            results['true_ate'].append(true_ate)
            results['bias'].append(matcher.ate_ - true_ate)
            results['percent_bias'].append(100 * (matcher.ate_ - true_ate) / true_ate)
            results['overlap_initial'].append(matcher.overlap_initial_)
            results['overlap_final'].append(matcher.overlap_final_)
            results['treatment_retention'].append(matcher.treatment_retention_)
            
            print(f"ATE: {matcher.ate_:.4f} (True: {true_ate:.4f})")
            print(f"Bias: {matcher.ate_ - true_ate:.4f}")
            
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")
            continue
    
    # Summarize results
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Successful iterations: {len(results_df)}")
    print(f"Mean ATE estimate: {results_df['estimated_ate'].mean():.4f}")
    print(f"True ATE: {results_df['true_ate'].mean():.4f}")
    print(f"Mean bias: {results_df['bias'].mean():.4f}")
    print(f"Mean % bias: {results_df['percent_bias'].mean():.2f}%")
    print(f"Std of ATE estimates: {results_df['estimated_ate'].std():.4f}")
    print(f"Mean overlap improvement: {results_df['overlap_initial'].mean():.4f} → {results_df['overlap_final'].mean():.4f}")
    print(f"Mean treatment retention: {results_df['treatment_retention'].mean():.2%}")
    print("="*60)
    
    return results_df

if __name__ == "__main__":
    # Run simulation
    results = run_simulation(n_iterations=10, n_samples=1000)
    
    # Optionally save results
    # results.to_csv('simulation_results.csv', index=False)
