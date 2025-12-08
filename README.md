# Remote Sensing Course - Code Repository

This repository contains code and analysis for the Remote Sensing course (Fall 2025) taught by [Dr. Parviz Fatehi](https://profile.ut.ac.ir/en/~parviz.fatehi).

**Author**: [Ali Haghighi](https://hellotherebeautiful.vercel.app/)

## Repository Structure

Each homework assignment is organized in its own folder:

```
├── homework_03/          # Land Surface Temperature (LST) Analysis
├── homework_04/          # Future assignments...
└── utils/                # Shared utilities (if any)
```

## Homework 3: Land Surface Temperature (LST) Analysis

**Objective**: Calculate and visualize Land Surface Temperature from Landsat 5 TM satellite imagery using Google Earth Engine.

### Key Features
- Natural RGB and False Color Composite visualization
- LST calculation from thermal Band 6
- Full scene and city-level (Rasht) temperature analysis
- Statistical analysis and histograms

### Data Source
- **Satellite**: Landsat 5 Thematic Mapper (TM)
- **Collection**: LANDSAT/LT05/C02/T1 (Collection 2, Tier 1)
- **Scene**: LT51660341987186XXX02 (July 5, 1987)
- **Location**: Path 166, Row 34 (Northern Iran)

### Files
- `homework_03_lst_analysis.ipynb` - Main analysis notebook
- `scripts/` - Supporting Python scripts for modular execution
- `data/selected_scene.json` - Scene metadata
- `output/` - Generated visualizations and maps

## Setup

### Prerequisites
- Python 3.8+
- Google Earth Engine account
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Remote Sensing Course"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Authenticate Google Earth Engine:
```python
import ee
ee.Authenticate()
ee.Initialize(project='your-project-name')
```

## Usage

### For Homework 3 (LST Analysis)

**Option 1: Use the Jupyter Notebook**
```bash
jupyter notebook homework_03/homework_03_lst_analysis.ipynb
```

**Option 2: Run Python Scripts**
```bash
cd homework_03/scripts
python load_landsat5_tm_band6.py      # Select scene
python calculate_lst.py                # Full LST analysis
python calculate_lst_celsius.py        # Celsius map with histogram
python calculate_lst_rasht.py          # Rasht city analysis
```

## Requirements

See `requirements.txt` for full list. Main dependencies:
- `earthengine-api` - Google Earth Engine Python API
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `Pillow` - Image processing

## Output

All generated visualizations and analysis outputs are saved in each homework's `output/` directory.

## License

Academic use - Course assignment repository

