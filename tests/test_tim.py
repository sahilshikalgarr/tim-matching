"""
Unit tests for TIM package
"""

import pytest
import numpy as np
import pandas as pd
from tim import TIMatcher

def generate_test_data(n=200):
    """Generate small test dataset"""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(0, 1, n),
        'x3': np.random.choice([0, 1], n),
        'treatment': np.random.choice([0, 1], n),
    })
    
    data['outcome'] = (
        2.0 * data['treatment'] +
        0.5 * data['x1'] +
        0.3 * data['x2'] +
        1.0 * data['x3'] +
        np.random.normal(0, 0.5, n)
    )
    
    return data

class TestTIMatcher:
    
    def test_initialization(self):
        """Test TIMatcher initialization"""
        matcher = TIMatcher(
            treatment_col='treatment',
            outcome_col='outcome',
            continuous_cols=['x1', 'x2'],
            discrete_cols=['x3']
        )
        assert matcher.treatment_col == 'treatment'
        assert matcher.outcome_col == 'outcome'
        assert len(matcher.continuous_cols) == 2
        assert len(matcher.discrete_cols) == 1
    
    def test_fit(self):
        """Test fitting matcher"""
        data = generate_test_data()
        
        matcher = TIMatcher(
            treatment_col='treatment',
            outcome_col='outcome',
            continuous_cols=['x1', 'x2'],
            discrete_cols=['x3'],
            coarsen_bins=3
        )
        
        matcher.fit(data)
        
        # Check that attributes are set
        assert matcher.matched_data_ is not None
        assert matcher.weights_ is not None
        assert matcher.ate_ is not None
        assert matcher.att_ is not None
        assert matcher.overlap_initial_ is not None
        assert matcher.overlap_final_ is not None
    
    def test_input_validation(self):
        """Test input validation"""
        data = generate_test_data()
        
        # Missing column
        matcher = TIMatcher(
            treatment_col='treatment',
            outcome_col='outcome',
            continuous_cols=['x1', 'x2', 'missing_col'],
            discrete_cols=['x3']
        )
        
        with pytest.raises(ValueError):
            matcher.fit(data)
    
    def test_get_matched_data(self):
        """Test getting matched data"""
        data = generate_test_data()
        
        matcher = TIMatcher(
            treatment_col='treatment',
            outcome_col='outcome',
            continuous_cols=['x1', 'x2'],
            discrete_cols=['x3']
        )
        
        matcher.fit(data)
        matched_data = matcher.get_matched_data()
        
        assert 'tim_weights' in matched_data.columns
        assert len(matched_data) <= len(data)

if __name__ == "__main__":
    pytest.main([__file__])
```

Commit: "Add unit tests"

---

## Method 2: Upload Using GitHub Desktop (Graphical Interface)

This is easier than command line but requires installing software.

### Step 1: Install GitHub Desktop

1. Download from: https://desktop.github.com/
2. Install and open
3. Sign in with your GitHub account

### Step 2: Create Repository

1. **In GitHub Desktop:**
   - Click "File" → "New Repository"
   - Name: `tim-matching`
   - Local path: Choose where to save on your computer
   - Check "Initialize this repository with a README"
   - Git ignore: Python
   - License: MIT
   - Click "Create Repository"

### Step 3: Add Your Files

1. **Open the repository folder** (GitHub Desktop shows the path)
2. **Copy all your Python files** into this folder
3. **Organize into the folder structure:**
```
   tim-matching/
   ├── tim/
   │   ├── __init__.py
   │   ├── matcher.py
   │   ├── importance.py
   │   └── ...
   ├── examples/
   └── ...
```

### Step 4: Commit and Push

1. **In GitHub Desktop:**
   - You'll see all changed files in the left sidebar
   - Bottom left: Add commit message "Initial commit: TIM library"
   - Click "Commit to main"
   - Click "Publish repository" button at top
   - Choose Public/Private
   - Click "Publish Repository"

Done! Your repository is now on GitHub.

---

## Method 3: Use Visual Studio Code with GitHub Extension

### Step 1: Install VS Code
1. Download from: https://code.visualstudio.com/
2. Install

### Step 2: Install GitHub Extension
1. Open VS Code
2. Click Extensions icon (left sidebar)
3. Search "GitHub"
4. Install "GitHub Pull Requests and Issues"

### Step 3: Sign In and Publish
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "GitHub: Publish to GitHub"
3. Follow prompts
4. Choose Public/Private
5. Select files to include
6. Done!

---

## After Upload: Important Next Steps

### 1. Edit Repository Settings

Go to your repository → Settings:

**About section (right sidebar):**
- Add description
- Add topics: `causal-inference`, `matching`, `python`, `statistics`
- Add website (if you have documentation)

**Features:**
- ✅ Issues (for bug reports)
- ✅ Discussions (for Q&A)

### 2. Add Topics/Tags

On your repository homepage:
- Click gear icon next to "About"
- Add topics:
  - `causal-inference`
  - `matching`
  - `statistics`
  - `machine-learning`
  - `python`
  - `treatment-effects`

### 3. Create First Release

1. Go to "Releases" (right sidebar)
2. Click "Create a new release"
3. Tag version: `v0.1.0`
4. Release title: "Initial Release v0.1.0"
5. Description:
```
   First public release of TIM (Two-Stage Interpretable Matching)
   
   Features:
   - Automated confounder importance calculation
   - CEM with variable prioritization
   - Unified distance metrics
   - ATE/ATT estimation
   
   Installation:
```bash
   pip install git+https://github.com/yourusername/tim-matching.git
```
