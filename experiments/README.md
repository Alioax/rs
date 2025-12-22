# Experiments & Explorations

This folder contains side experiments, explorations, and experimental code related to remote sensing but outside the scope of official homework assignments.

## Purpose

- **Testing new methods** before using them in homework
- **Exploring datasets** beyond assignment requirements
- **Comparing different approaches** (e.g., LST calculation methods)
- **Trying new techniques** (e.g., machine learning, advanced indices)
- **Personal learning** and skill development

## Organization

Each experiment should follow a similar structure to homework folders:

```
experiments/
├── exp_[descriptive_name]/
│   ├── exp_[name].ipynb          # Main notebook
│   ├── README.md                  # What this experiment does
│   ├── scripts/                   # Supporting scripts (optional)
│   ├── data/                      # Input data (optional)
│   └── output/                    # Results/outputs
```

## Naming Convention

Use descriptive names with `exp_` prefix:
- `exp_lst_comparison` - Comparing different LST calculation methods
- `exp_ndvi_analysis` - NDVI exploration
- `exp_sentinel2_lst` - Testing Sentinel-2 for LST
- `exp_landsat_comparison` - Comparing Landsat 5 vs 8
- `exp_urban_heat_island` - Urban heat island analysis

## Examples

### Example 1: LST Method Comparison
```
exp_lst_comparison/
├── exp_lst_comparison.ipynb
├── README.md
└── output/
    ├── method1_results.png
    └── comparison_plot.png
```

### Example 2: Quick Test
```
exp_sentinel2_test/
└── exp_sentinel2_test.ipynb  # Simple notebook, no subfolders needed
```

## Notes

- **No strict structure required** - experiments can be as simple or complex as needed
- **Documentation encouraged** - Add a README explaining what you're testing
- **Feel free to experiment** - This is your space to explore!

