# TIM: Two-Stage Interpretable Matching

A Python library for causal inference using Two-Stage Interpretable Matching (TIM), which combines covariate importance weighting with exact matching for robust treatment effect estimation.

## Features

- Automated confounder importance calculation
- Coarsened Exact Matching (CEM) with variable prioritization
- Unified distance metrics for mixed continuous/discrete covariates
- Inverse distance weighting for improved balance
- ATE and ATT estimation with proper weights

## Installation
```bash
pip install tim-matching
```

Or install from source:
```bash
git clone https://github.com/yourusername/tim-matching.git
cd tim-matching
pip install -e .
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
print(f"ATE: {matcher.ate_}")
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
@misc{tim2024,
  title={Two-Stage Interpretable Matching for Causal Inference},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/tim-matching}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Author: Your Name
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

## Acknowledgments

Based on research from [your institution/lab].
```

5. Scroll to bottom
6. Add commit message: "Update README"
7. Click **"Commit changes"**

#### 2. Create `requirements.txt`

1. Click **"Add file" → "Create new file"**
2. Name it: `requirements.txt`
3. Paste this content:
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
cem>=0.1.0
