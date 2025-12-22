# Side Experiments Guide

## Overview

The `experiments/` folder is your space for side projects, explorations, and experimental code that's related to remote sensing but outside the scope of official homework assignments.

## When to Use Experiments Folder

✅ **Use `experiments/` for:**
- Testing new methods before using in homework
- Comparing different approaches (e.g., LST calculation methods)
- Exploring datasets beyond assignment requirements
- Trying new techniques (machine learning, advanced indices)
- Personal learning and skill development
- Quick tests and prototypes

❌ **Keep in `homework_XX/` for:**
- Official course assignments
- Required deliverables
- Graded work

## Quick Start

### Option 1: Simple Experiment (Quick Test)
```bash
mkdir experiments/exp_lst_emissivity_test
# Create your notebook: exp_lst_emissivity_test.ipynb
```

### Option 2: Structured Experiment
```bash
mkdir -p experiments/exp_lst_comparison/{scripts,data,output}
# Create notebook and README
```

## Example Use Cases

### 1. Method Comparison
**Folder**: `experiments/exp_lst_method_comparison/`
**Purpose**: Compare different LST calculation methods
**Files**: 
- `exp_lst_method_comparison.ipynb`
- `output/comparison_results.png`

### 2. Dataset Exploration
**Folder**: `experiments/exp_sentinel2_exploration/`
**Purpose**: Explore Sentinel-2 data capabilities
**Files**: 
- `exp_sentinel2_exploration.ipynb`
- `README.md` (explaining findings)

### 3. Advanced Technique Test
**Folder**: `experiments/exp_ml_classification/`
**Purpose**: Test machine learning for land cover classification
**Files**: 
- `exp_ml_classification.ipynb`
- `scripts/preprocessing.py`
- `output/classification_results/`

## Structure Flexibility

Experiments can be as simple or complex as needed:

**Minimal** (for quick tests):
```
exp_quick_test/
└── exp_quick_test.ipynb
```

**Standard** (recommended):
```
exp_lst_comparison/
├── exp_lst_comparison.ipynb
├── README.md
└── output/
```

**Full Structure** (for complex experiments):
```
exp_advanced_analysis/
├── exp_advanced_analysis.ipynb
├── README.md
├── scripts/
├── data/
└── output/
```

## Naming Convention

Use descriptive names with `exp_` prefix:
- ✅ `exp_lst_emissivity_comparison`
- ✅ `exp_sentinel2_ndvi`
- ✅ `exp_landsat5_vs_landsat8`
- ❌ `test1` or `experiment` (too vague)

## Benefits

1. **Separation**: Keeps official homework clean and organized
2. **Exploration**: Safe space to test ideas without affecting assignments
3. **Learning**: Document your learning journey
4. **Portfolio**: Shows initiative and exploration beyond coursework
5. **Flexibility**: No strict structure required - adapt to your needs

## Tips

- **Document your experiments**: Add a README explaining what you're testing
- **Use descriptive names**: Future you will thank you
- **Keep it organized**: Even experiments benefit from some structure
- **Learn freely**: This is your space to explore and make mistakes!

## Example Workflow

```bash
# 1. Create experiment folder
mkdir experiments/exp_lst_emissivity_test
cd experiments/exp_lst_emissivity_test

# 2. Create notebook
jupyter notebook exp_lst_emissivity_test.ipynb

# 3. (Optional) Add README explaining what you're testing
# 4. (Optional) Create output folder for results
mkdir output

# 5. Experiment and learn! 🚀
```

---

**Remember**: Experiments are for exploration and learning. Don't worry about perfection - just explore and document what you learn!

