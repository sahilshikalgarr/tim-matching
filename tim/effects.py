"""
Treatment effect estimation
"""


def calculate_ate_att(matched_dfs, treatment, outcome, weight):
    """
    Calculate Average Treatment Effect (ATE) and Average Treatment Effect on Treated (ATT).
    
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
    tuple
        (final_ate, final_att)
    """
    ate_list = []
    att_list = []
    
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
            
            att = ((treated[outcome] * treated[weight]).sum() / treated[weight].sum() -
                  (control[outcome] * control[weight]).sum() / control[weight].sum())
            att_list.append(att)
    
    final_ate = sum(ate_list) / len(ate_list) if ate_list else None
    final_att = sum(att_list) / len(att_list) if att_list else None
    
    return final_ate, final_att
