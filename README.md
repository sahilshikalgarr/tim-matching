# TIM: Two-Stage Interpretable Matching

A Python library for causal inference using Two-Stage Interpretable Matching (TIM), which combines covariate importance weighting with exact matching for robust treatment effect estimation.

## Features

- Automated confounder importance calculation
- Coarsened Exact Matching (CEM) with variable prioritization
- Unified distance metrics for mixed continuous/discrete covariates
- Inverse distance weighting for improved balance
- CATE estimation with proper weights

## Installation
```bash
%pip install git+https://github.com/sahilshikalgarr/tim-matching.git
```

## Quick Start
```python
import pandas as pd
from tim import TIMatcher

# Load your data
data = pd.read_csv('your_data.csv')

# Initialize matcher
matcher = TIMatcher(
    treatment_col='treatment',
    outcome_col='outcome',
    continuous_cols=['age', 'income', 'bmi'],
    discrete_cols=['gender', 'education'],
    coarsen_bins=4
)

# Fit the matcher
matcher.fit(data)

# View results
matcher.summary()

# Get matched data with weights
matched_data = matcher.get_matched_data()

# Access treatment effects
print(f"CATE: {matcher.ate_}")
```

## Examples

See the `examples/` directory for:
- Basic usage examples
- Simulation studies
- Real-world applications

## Requirements

- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- statsmodels >= 0.13.0
- cem >= 0.1.0

## Citation

If you use this library in your research, please cite:
```bibtex
@article{shikalgar2025two,
  title={A two-stage interpretable matching framework for causal inference},
  author={Shikalgar, Sahil and Noor-E-Alam, Md},
  journal={IISE Transactions on Healthcare Systems Engineering},
  pages={1--14},
  year={2025},
  publisher={Taylor \& Francis}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

- Author: Sahil Shikalgar
- Email: shikalgar.s@northeastern.edu

## Acknowledgments

Based on research from Decision Analytics Lab at Northeastern University.
```


