"""
Treatment effect estimation
"""
def calculate_ate(matched_dfs, treatment, outcome, weight):
    """
    Calculate Conditional Average Treatment Effect (CATE).
    
    Parameters
    ----------
    matched_dfs : list of tuples
        List of (covariates, matched_df) tuples
    treatment : str
        Treatment column name
    outcome : str
        Outcome column name
    weight : str
        Weight column name
        
    Returns
    -------
    float
       Conditional Average Treatment Effect (CATE)
    """
    ate_list = []
    
    for covariates, matched_df in matched_dfs:
        if {treatment, outcome, weight}.issubset(matched_df.columns):
            treated = matched_df[matched_df[treatment] == 1]
            control = matched_df[matched_df[treatment] == 0]
            
            n_treat = treated.shape[0]
            n_control = control.shape[0]
            
            treated_effect = treated[outcome].sum() / n_treat
            
            if "inverse_distance" in matched_df.columns:
                control_effect = ((control[outcome] * control["inverse_distance"]).sum() / 
                                control["inverse_distance"].sum())
            else:
                control_effect = control[outcome].sum() / n_control
            
            ate = treated_effect - control_effect
            
            for _ in range(n_treat):
                ate_list.append(ate)
    
    final_ate = sum(ate_list) / len(ate_list) if ate_list else None
    
    return final_ate
