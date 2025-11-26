"""
Main TIM matcher class
"""

import numpy as np
import pandas as pd
import warnings
from cem.match import match
from cem.coarsen import coarsen
from cem.imbalance import L1

from .importance import confounder_importance_conti
from .distances import algo_distance_crosstab, unified_distance
from .weights import calculate_weights_from_best_matches_inverse_append
from .effects import calculate_ate_att

warnings.filterwarnings('ignore', category=FutureWarning)


class TIMatcher:
    """
    Two-Stage Interpretable Matching (TIM) for causal inference.
    
    Parameters
    ----------
    treatment_col : str
        Name of the treatment column (binary: 0 for control, 1 for treated)
    outcome_col : str
        Name of the outcome column
    continuous_cols : list
        List of continuous covariate column names
    discrete_cols : list
        List of discrete covariate column names
    coarsen_bins : int, optional (default=4)
        Number of bins for coarsening continuous variables
    
    Attributes
    ----------
    matched_data_ : pd.DataFrame
        Matched dataset after fitting
    weights_ : pd.Series
        Weights for matched units
    ate_ : float
        Average Treatment Effect
    att_ : float
        Average Treatment Effect on the Treated
    overlap_initial_ : float
        Initial overlap metric (L1)
    overlap_final_ : float
        Final overlap metric after matching (L1)
    """
    
    def __init__(self, treatment_col, outcome_col, continuous_cols, 
                 discrete_cols, coarsen_bins=4):
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.continuous_cols = continuous_cols
        self.discrete_cols = discrete_cols
        self.coarsen_bins = coarsen_bins
        
        # Attributes set after fitting
        self.matched_data_ = None
        self.weights_ = None
        self.ate_ = None
        self.att_ = None
        self.overlap_initial_ = None
        self.overlap_final_ = None
        self.matched_strata_ = None
        self.confounder_importance_ = None
        
    def fit(self, data):
        """
        Fit the TIM matcher to the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe containing treatment, outcome, and covariates
            
        Returns
        -------
        self : TIMatcher
            Fitted matcher object
        """
        # Validate input
        self._validate_input(data)
        
        # Store original data
        self.data_ = data.copy()
        
        # Calculate initial overlap
        initial_data = data.drop([self.outcome_col], axis=1)
        self.overlap_initial_ = L1(initial_data, self.treatment_col)
        
        # Step 1: Calculate confounder importance
        print("Calculating confounder importance...")
        self.confounder_importance_ = confounder_importance_conti(
            data, self.outcome_col, self.treatment_col
        )
        
        # Step 2: Coarsen continuous variables
        print("Coarsening continuous variables...")
        X_coarse = coarsen(
            data, 
            self.treatment_col, 
            "l1", 
            lower=self.coarsen_bins,
            columns=self.continuous_cols
        )
        
        # Step 3: Perform exact matching with variable importance
        print("Performing exact matching...")
        all_covariates = self.continuous_cols + self.discrete_cols
        matched_strata, unmatched_treated, unmatched_control, matched_df = \
            self._exact_matching_with_importance(
                X_coarse, 
                treatment_col=self.treatment_col,
                covariate_cols=all_covariates,
                variable_importance=self.confounder_importance_
            )
        
        self.matched_strata_ = matched_strata
        self.matched_data_ = matched_df
        
        # Step 4: Calculate discrete distances
        print("Calculating distances...")
        if self.discrete_cols:
            disc_dist = algo_distance_crosstab(
                df=X_coarse, 
                disc_columns=self.discrete_cols,
                treatment_col=self.treatment_col,
                outcome_col=self.outcome_col
            )
            
            # Calculate unified distance
            unified_distance(
                matched_dfs=matched_strata,
                treatment=self.treatment_col,
                outcome=self.outcome_col,
                continuous_cols=self.continuous_cols,
                disc_distance=disc_dist,
                matched_list=all_covariates,
                data=data
            )
        
        # Step 5: Calculate weights
        print("Calculating weights...")
        _, self.weights_ = calculate_weights_from_best_matches_inverse_append(
            matched_strata, 
            self.treatment_col
        )
        
        # Step 6: Calculate treatment effects
        print("Calculating treatment effects...")
        self.ate_, self.att_ = calculate_ate_att(
            matched_dfs=matched_strata,
            treatment=self.treatment_col,
            outcome=self.outcome_col,
            weight='weights'
        )
        
        # Step 7: Calculate final overlap
        final_df = matched_df.drop(self.outcome_col, axis=1)
        self.overlap_final_ = L1(final_df, self.treatment_col, self.weights_)
        
        # Calculate treatment retention
        self.treatment_retention_ = (
            matched_df[matched_df[self.treatment_col] == 1].shape[0] /
            data[data[self.treatment_col] == 1].shape[0]
        )
        
        print("Matching complete!")
        return self
    
    def _validate_input(self, data):
        """Validate input data"""
        required_cols = [self.treatment_col, self.outcome_col] + \
                       self.continuous_cols + self.discrete_cols
        
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        # Check treatment is binary
        unique_treatment = data[self.treatment_col].nunique()
        if unique_treatment != 2:
            raise ValueError(
                f"Treatment column must be binary. Found {unique_treatment} unique values."
            )
    
    def _exact_matching_with_importance(self, df, treatment_col, covariate_cols, 
                                       variable_importance):
        """
        Perform exact matching using variable importance ranking.
        """
        # Sort covariates by importance
        sorted_covariates = sorted(
            variable_importance.keys(), 
            key=lambda x: variable_importance[x], 
            reverse=True
        )
        
        remaining_df = df.copy()
        treatment_levels = df[treatment_col].nunique()
        strata_list = []
        
        # Iterate from all covariates down to single covariate
        for subset_size in range(len(sorted_covariates), 0, -1):
            current_covariates = sorted_covariates[:subset_size]
            
            # Group by current strata
            grouped = remaining_df.groupby(current_covariates, observed=True)
            
            # Find valid strata (containing both treatment and control)
            for key, group in grouped:
                if group[treatment_col].nunique() == treatment_levels:
                    strata_list.append((current_covariates, pd.DataFrame(group)))
                    indices = pd.DataFrame(group).index
                    remaining_df = remaining_df.drop(index=indices, errors='ignore')
            
            # Check if all treated units are matched
            remaining_treated = remaining_df[remaining_df[treatment_col] == 1]
            if remaining_treated.empty:
                break
        
        # Consolidate matched data
        matched_df_final = pd.concat([match[1] for match in strata_list]) \
                          if strata_list else pd.DataFrame()
        
        remaining_treated = remaining_df[remaining_df[treatment_col] == 1]
        
        return strata_list, remaining_treated, remaining_df, matched_df_final
    
    def summary(self):
        """
        Print a summary of matching results.
        """
        if self.matched_data_ is None:
            raise ValueError("Matcher has not been fitted yet. Call fit() first.")
        
        print("=" * 60)
        print("TIM Matching Summary")
        print("=" * 60)
        print(f"Treatment column: {self.treatment_col}")
        print(f"Outcome column: {self.outcome_col}")
        print(f"Continuous covariates: {len(self.continuous_cols)}")
        print(f"Discrete covariates: {len(self.discrete_cols)}")
        print()
        print("Sample Sizes:")
        print(f"  Original treated: {self.data_[self.data_[self.treatment_col]==1].shape[0]}")
        print(f"  Original control: {self.data_[self.data_[self.treatment_col]==0].shape[0]}")
        print(f"  Matched treated: {self.matched_data_[self.matched_data_[self.treatment_col]==1].shape[0]}")
        print(f"  Matched control: {self.matched_data_[self.matched_data_[self.treatment_col]==0].shape[0]}")
        print(f"  Treatment retention: {self.treatment_retention_:.2%}")
        print()
        print("Balance:")
        print(f"  Initial overlap (L1): {self.overlap_initial_:.4f}")
        print(f"  Final overlap (L1): {self.overlap_final_:.4f}")
        print()
        print("Treatment Effects:")
        print(f"  ATE: {self.ate_:.4f}")
        print(f"  ATT: {self.att_:.4f}")
        print("=" * 60)
    
    def get_matched_data(self):
        """
        Get the matched dataset.
        
        Returns
        -------
        pd.DataFrame
            Matched data with weights
        """
        if self.matched_data_ is None:
            raise ValueError("Matcher has not been fitted yet. Call fit() first.")
        
        matched_with_weights = self.matched_data_.copy()
        matched_with_weights['tim_weights'] = self.weights_
        return matched_with_weights
