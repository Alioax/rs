# Remote Sensing Course Repository Structure

## Proposed Organization

```
Remote Sensing Course/
├── README.md                          # Main repository documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore file
│
├── homework_01/                      # Homework 1 (if applicable)
│   ├── homework_01.ipynb
│   ├── scripts/                      # Supporting Python scripts (optional)
│   ├── data/                         # Input data files
│   └── output/                       # Generated outputs
│
├── homework_02/                      # Homework 2 (if applicable)
│   ├── homework_02.ipynb
│   ├── scripts/
│   ├── data/
│   └── output/
│
├── homework_03/                      # Homework 3: LST Analysis
│   ├── homework_03_lst_analysis.ipynb
│   ├── scripts/                      # Supporting scripts
│   │   ├── load_landsat5_tm_band6.py
│   │   ├── calculate_lst.py
│   │   ├── calculate_lst_celsius.py
│   │   └── calculate_lst_rasht.py
│   ├── data/
│   │   └── selected_scene.json
│   └── output/                        # All outputs for this homework
│       ├── RGB_*.png
│       ├── LST_*.png
│       └── ...
│
├── homework_04/                      # Future homework
│   ├── homework_04.ipynb
│   ├── scripts/
│   ├── data/
│   └── output/
│
├── experiments/                      # Side experiments & explorations
│   ├── exp_lst_comparison/           # Example: Compare LST methods
│   │   ├── exp_lst_comparison.ipynb
│   │   ├── scripts/
│   │   ├── data/
│   │   └── output/
│   ├── exp_ndvi_analysis/            # Example: NDVI exploration
│   │   └── ...
│   └── exp_sentinel2_lst/            # Example: Sentinel-2 LST test
│       └── ...
│
└── utils/                            # Shared utilities (optional)
    ├── gee_helpers.py                # Common GEE functions
    └── visualization.py              # Common plotting functions
```

## Benefits of This Structure

1. **Clear Separation**: Each homework is self-contained in its own folder
2. **Scalable**: Easy to add new homeworks without cluttering
3. **Organized Outputs**: Each homework's outputs stay with its code
4. **Flexible**: Can include supporting scripts alongside notebooks
5. **Professional**: Clean structure suitable for portfolio/GitHub
6. **Experiments**: Side projects are clearly separated from official homework
7. **Exploratory**: Safe space for testing ideas without mixing with assignments

## Alternative: Simpler Structure

If you prefer a simpler approach:

```
Remote Sensing Course/
├── README.md
├── requirements.txt
├── homework_03_lst_analysis.ipynb    # Main notebook
├── homework_03_scripts/              # Supporting scripts
├── homework_03_outputs/               # Outputs
└── data/                              # Shared data (if any)
```

## Recommendation

**Go with the first structure (homework_XX folders)** because:
- You'll have multiple homeworks
- Keeps everything organized
- Easy to navigate
- Professional appearance
- Each homework can have its own README if needed

