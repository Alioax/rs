# Experiment Template

Use this as a guide when creating new experiments.

## Quick Start

1. Create a new folder: `experiments/exp_[descriptive_name]/`
2. Add your notebook: `exp_[name].ipynb`
3. Optionally add a README explaining what you're testing

## Example Structure

### Minimal (for quick tests)
```
exp_quick_test/
└── exp_quick_test.ipynb
```

### Standard (recommended)
```
exp_lst_method_comparison/
├── exp_lst_method_comparison.ipynb
├── README.md
└── output/
    └── comparison_results.png
```

### Full Structure (for complex experiments)
```
exp_advanced_analysis/
├── exp_advanced_analysis.ipynb
├── README.md
├── scripts/
│   ├── helper_functions.py
│   └── data_processing.py
├── data/
│   └── input_data.csv
└── output/
    ├── results/
    └── figures/
```

## README Template

```markdown
# Experiment: [Name]

## Purpose
What are you testing/exploring?

## Hypothesis/Question
What question are you trying to answer?

## Methods
- Dataset: [What data?]
- Approach: [What method/technique?]
- Comparison: [What are you comparing?]

## Results
Brief summary of findings

## Notes
Any observations, issues, or next steps
```

## Naming Tips

- Use descriptive names: `exp_lst_comparison` not `exp_test1`
- Be specific: `exp_sentinel2_ndvi` not `exp_ndvi`
- Use underscores: `exp_urban_heat_island` not `expUrbanHeatIsland`

## Examples

- `exp_lst_emissivity_comparison` - Testing different emissivity values
- `exp_landsat5_vs_landsat8` - Comparing Landsat versions
- `exp_ndvi_calculation` - Testing NDVI calculation methods
- `exp_sentinel2_lst` - Exploring Sentinel-2 for LST
- `exp_machine_learning_classification` - ML classification test

