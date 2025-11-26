"""
Distance calculation functions
"""

import numpy as np
import pandas as pd
from itertools import combinations


def find_max_crosstab(col1, col2, attribute1, attribute2):
    """
    Calculate distance between two attributes across categories.
    """
    data = {'c1': list(col1), 'c2': list(col2)}
    df = pd.DataFrame(data)
    
    attribute1_joint_counts = pd.crosstab(index=df['c1'], columns=df['c2'])
    
    distance = 0
    for i in df['c2'].unique():
        prob_attr1 = (attribute1_joint_counts[i].loc[attribute1] / 
                     attribute1_joint_counts.loc[attribute1].sum())
        prob_attr2 = (attribute1_joint_counts[i].loc[attribute2] / 
                     attribute1_joint_counts.loc[attribute2].sum())
        
        if prob_attr1 >= prob_attr2:
            distance += prob_attr1
        else:
            distance += prob_attr2
    
    distance = distance - 1
    return distance


def algo_distance_crosstab(df, disc_columns, treatment_col, outcome_col):
    """
    Calculate pairwise distances for discrete variables.
    """
    df = df.drop([outcome_col, treatment_col], axis=1)
    cols = disc_columns
    results = []
    
    for i in cols:
        for current_combination in combinations(df[i].unique(), 2):
            total = 0
            for j in cols:
                if i != j:
                    dist = find_max_crosstab(
                        col1=df[i], 
                        col2=df[j],
                        attribute1=current_combination[0],
                        attribute2=current_combination[1]
                    )
                    total += dist
            
            total_dist = total / (len(cols) - 1) if len(cols) > 1 else 0
            
            results.append({
                'Column_Name': i,
                'Attribute_1': current_combination[0],
                'Attribute_2': current_combination[1],
                'Total_Distance': total_dist
            })
    
    results_df = pd.DataFrame(results)
    return results_df


def unified_distance(matched_dfs, treatment, outcome, continuous_cols,
                    disc_distance, matched_list, data):
    """
    Calculate unified distance incorporating both continuous and discrete variables.
    """
    for covariates, matched_df in matched_dfs:
        cov = covariates
        matched = matched_df.drop(cov, axis=1)
        
        common_values = list(set(matched.columns).intersection(set(continuous_cols)))
        df_merged = pd.merge(
            matched, 
            data[common_values], 
            left_index=True, 
            right_index=True,
            suffixes=('', '_continuous')
        )
        
        if any(column in matched_list for column in df_merged.columns):
            continuous_list = [i for i in df_merged.columns if 'continuous' in i]
            disc = [i for i in df_merged.columns if 'continuous' not in i]
            disc.remove(outcome)
            disc.remove(treatment)
            discrete_list = [item for item in disc 
                           if not any(item in word for word in continuous_list)]
            
            # Initialize distance columns
            df_merged['continuous_distance'] = np.nan
            df_merged['discrete_distance'] = np.nan
            
            num_discrete = len(discrete_list)
            num_continuous = len(continuous_list)
            total_variables = len(matched_list)
            inverse_weight = 1 - (num_discrete + num_continuous) / total_variables
            
            # Calculate continuous distances
            total_dist = [0] * df_merged[df_merged[treatment] == 0][continuous_list].shape[0]
            
            for i in continuous_list:
                training_1 = df_merged[df_merged[treatment] == 1][i].values
                training_0 = df_merged[df_merged[treatment] == 0][i].values
                distances = [np.min(np.abs(val - training_1)) for val in training_0]
                total_dist = [sum(i) for i in zip(total_dist, distances)]
            
            df_merged.loc[df_merged[treatment] == 0, 'continuous_distance'] = total_dist
            
            # Calculate discrete distances
            if discrete_list:
                total_dist = [0] * df_merged[df_merged[treatment] == 0][discrete_list].shape[0]
                
                for i in discrete_list:
                    treatment_element = df_merged[i].iloc[0]
                    distances = []
                    
                    for j, k in df_merged[df_merged[treatment] == 0].iterrows():
                        control_element = df_merged[i].loc[j]
                        row = disc_distance[
                            ((disc_distance['Attribute_1'] == treatment_element) & 
                             (disc_distance['Attribute_2'] == control_element)) |
                            ((disc_distance['Attribute_1'] == control_element) & 
                             (disc_distance['Attribute_2'] == treatment_element))
                        ]
                        dist = row["Total_Distance"].values[0] if not row.empty else 0
                        distances.append(dist)
                    
                    total_dist = [sum(i) for i in zip(total_dist, distances)]
                
                df_merged.loc[df_merged[treatment] == 0, 'discrete_distance'] = total_dist
            
            # Calculate grand total
            df_merged['grand_total'] = (df_merged['continuous_distance'].fillna(0) + 
                                       df_merged['discrete_distance'].fillna(0))
            df_merged.loc[df_merged[treatment] == 1, 'grand_total'] = np.nan
            
            # Inverse Min-Max Normalization
            min_value = df_merged[df_merged[treatment] == 0]['grand_total'].min()
            max_value = df_merged[df_merged[treatment] == 0]['grand_total'].max()
            
            if max_value > min_value:
                df_merged['inverse_distance'] = (
                    1 - (df_merged['grand_total'] - min_value) / (max_value - min_value)
                )
            else:
                df_merged['inverse_distance'] = 1
            
            df_merged.loc[(df_merged[treatment] == 0) & 
                         (df_merged['inverse_distance'].isna()), 'inverse_distance'] = 1
            df_merged.loc[df_merged[treatment] == 1, 'inverse_distance'] = inverse_weight
            
            # Add columns to matched_df
            matched_df['discrete_distance'] = df_merged['discrete_distance']
            matched_df['continuous_distance'] = df_merged['continuous_distance']
            matched_df['grand_total'] = df_merged['grand_total']
            matched_df['inverse_distance'] = df_merged['inverse_distance']
