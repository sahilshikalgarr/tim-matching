"""
Confounder importance calculation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge


def confounder_importance_conti(df, outcome_col, treatment_col):
    """
    Calculate confounder importance for continuous outcomes using Ridge regression.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    outcome_col : str
        Outcome column name
    treatment_col : str
        Treatment column name
        
    Returns
    -------
    pd.Series
        Normalized confounder importance scores
    """
    df_out = df.copy()
    
    # Outcome association
    X_out = df_out.drop([outcome_col], axis=1)
    y_out = df_out[outcome_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X_out, y_out, train_size=0.7, random_state=42
    )
    
    # Hyperparameter tuning for outcome
    ridge = Ridge(random_state=43)
    params = {'alpha': [0.1, 1.0, 10, 100]}
    GS = GridSearchCV(
        estimator=ridge, 
        param_grid=params, 
        cv=10, 
        n_jobs=-1, 
        verbose=False, 
        scoring='r2'
    )
    GS.fit(X_train, y_train)
    best_ridge = Ridge(alpha=GS.best_params_["alpha"], random_state=43)
    best_ridge.fit(X_train, y_train)
    
    # Extract outcome coefficients
    outcome_coefficients = pd.Series(best_ridge.coef_, index=X_out.columns)
    outcome_coefficients = np.abs(outcome_coefficients)
    
    # Treatment association
    X_out = df_out.drop([treatment_col, outcome_col], axis=1)
    y_out = df_out[treatment_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X_out, y_out, train_size=0.7, random_state=42
    )
    
    GS.fit(X_train, y_train)
    best_ridge.fit(X_train, y_train)
    
    # Extract treatment coefficients
    treatment_coefficients = pd.Series(best_ridge.coef_, index=X_out.columns)
    treatment_coefficients = np.abs(treatment_coefficients)
    
    # Calculate confounder importance
    confounder_importance = abs(outcome_coefficients + treatment_coefficients)
    
    # Normalize
    max_importance = confounder_importance.max()
    if max_importance != 0:
        confounder_importance = confounder_importance / max_importance
    else:
        confounder_importance = pd.Series(
            np.zeros_like(confounder_importance), 
            index=confounder_importance.index
        )
    
    confounder_importance = confounder_importance.drop(treatment_col)
    return confounder_importance
