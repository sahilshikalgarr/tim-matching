"""
Weight calculation functions
"""

import pandas as pd


def calculate_weights_from_best_matches_inverse_append(best_matches, treatment_col):
    """
    Calculate and append weights for matched units.
    
    Parameters
    ----------
    best_matches : list of tuples
        List of (covariates, matched_df) tuples
    treatment_col : str
        Treatment column name
        
    Returns
    -------
    tuple
        (best_matches with weights, concatenated weight series)
    """
    # Global counts
    M_treated = sum([len(match_df[match_df[treatment_col] == 1]) 
                    for _, match_df in best_matches])
    M_control = sum([len(match_df[match_df[treatment_col] == 0]) 
                    for _, match_df in best_matches])
    
    weight_series_list = []
    
    for covariate_combination, match_df in best_matches:
        treated_units = match_df[match_df[treatment_col] == 1]
        control_units = match_df[match_df[treatment_col] == 0]
        
        n_stratum_treated = len(treated_units)
        n_stratum_control = len(control_units)
        
        match_df["weights"] = float('nan')
        
        # Treated weights
        match_df.loc[treated_units.index, "weights"] = 1
        treated_weights = pd.Series(1, index=treated_units.index)
        
        # Adjust by inverse_distance if exists
        if 'inverse_distance' in match_df.columns:
            treated_units_inv = match_df.loc[treated_units.index, 'inverse_distance']
            match_df.loc[treated_units.index, "weights"] *= treated_units_inv
            treated_weights *= treated_units_inv
        
        weight_series_list.append(treated_weights)
        
        # Control weights
        if n_stratum_treated > 0 and n_stratum_control > 0:
            weight_control = (M_control / M_treated) * (n_stratum_treated / n_stratum_control)
            
            if 'inverse_distance' in match_df.columns:
                control_units_inv = match_df.loc[control_units.index, 'inverse_distance']
                match_df.loc[control_units.index, "weights"] = weight_control * control_units_inv
                control_weights = pd.Series(weight_control, index=control_units.index) * control_units_inv
            else:
                match_df.loc[control_units.index, "weights"] = weight_control
                control_weights = pd.Series(weight_control, index=control_units.index)
            
            weight_series_list.append(control_weights)
    
    weights = pd.concat(weight_series_list)
    return best_matches, weights
