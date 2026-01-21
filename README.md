# Remote Sensing Course - Code Repository

This repository contains code and analysis for the Remote Sensing course (Fall 2025) taught by [Dr. Parviz Fatehi](https://profile.ut.ac.ir/en/~parviz.fatehi).

**Author**: [Ali Haghighi](https://hellotherebeautiful.vercel.app/)

## Repository Structure

The repository is organized into official homework assignments and side experiments:

```
├── homework_01/          # Data download (Landsat 8 Collection 2 Level 1)
├── homework_02/          # ROI selection and color composites
├── homework_03/          # Land Surface Temperature (LST) analysis
├── homework_04/          # NDVI-LST correlation analysis
├── homework_05/          # Geometric correction and resampling
├── homework_06/          # Scene search and image registration
├── homework_07/          # Change detection and diffusion comparison
├── homework_08/          # Unsupervised and supervised classification
├── experiments/          # Side experiments & explorations
│   └── exp_*/            # Individual experiments
└── utils/                # Shared utilities (if any)
```

### Official Homework (`homework_XX/`)
Structured assignments for the course. Each includes:
- Main analysis notebook or scripts
- Supporting scripts
- Data files
- Output visualizations
- Comprehensive README documentation

### Side Experiments (`experiments/exp_*/`)
Personal explorations, method comparisons, and experimental code related to remote sensing but outside official homework scope.

## Homework Assignments Overview

### Homework 1: Landsat 8 Data Download
**Objective**: Download Landsat 8 Collection 2 Level 1 satellite imagery from USGS EarthExplorer.

- **Data Source**: USGS EarthExplorer
- **Path/Row**: 166/34 (North Iran Region)
- **Purpose**: Data acquisition exercise (no processing required)
- **Used in**: Homework 2 for ROI selection and color composite visualization

### Homework 2: Selecting AOI and Color Composites
**Objective**: Select a Region of Interest (ROI) and visualize it using different color composite combinations.

- **Data Source**: Landsat 8 OLI (bands 2-7) from Homework 1
- **Key Features**:
  - Configurable rectangular ROI selection
  - Multiple color composite visualization (432, 543, 436, etc.)
  - Clean visualization without axes/grid for publication-ready output

### Homework 3: Land Surface Temperature (LST) Analysis
**Objective**: Calculate and visualize Land Surface Temperature from Landsat 5 TM satellite imagery using Google Earth Engine.

- **Data Source**: Landsat 5 TM, Path 166/Row 34 (Northern Iran)
- **Key Features**:
  - Natural RGB and False Color Composite visualization
  - LST calculation from thermal Band 6
  - Full scene and city-level (Rasht) temperature analysis
  - Statistical analysis and histograms

### Homework 4: NDVI-LST Correlation Analysis
**Objective**: Analyze the relationship between vegetation cover (NDVI) and land surface temperature (LST).

- **Data Source**: Same Landsat 5 TM image as Homework 3
- **Key Features**:
  - NDVI calculation from TOA reflectance
  - Vegetation masking (NDVI > 0.35)
  - Correlation analysis between NDVI and LST
  - Scatter plots and statistical analysis

### Homework 5: Geometric Correction and Resampling
**Objective**: Perform geometric correction using ground control points and compare resampling methods.

- **Data Source**: Unrectified satellite images (Shiraz, Iran)
- **Key Features**:
  - GCP-based geometric correction
  - RMSE calculation for accuracy assessment
  - Resampling method comparison (Nearest Neighbor, Bilinear, Cubic)
  - Road network overlay for verification

### Homework 6: Scene Search and Image Registration
**Objective**: Search Landsat archives and perform image-to-image registration.

- **Data Source**: Landsat 8/9 Collection 2, Path 162/Row 40
- **Key Features**:
  - Scene search with filtering criteria
  - Image-to-image registration (automatic and manual GCPs)
  - SLC error comparison and gap-fill methods

### Homework 7: Change Detection and Diffusion Comparison
**Objective**: Detect changes between 2015 and 2021 using Landsat imagery and compare resolution enhancement.

- **Data Source**: Landsat 8 OLI, Path 166/Row 34 (Northern Iran)
- **Key Features**:
  - Temporal change detection (2015 vs 2021)
  - Surface reflectance conversion
  - Pixel-wise difference analysis
  - Diffusion comparison (30m vs 15m resolution)
  - AOI-based visualization

### Homework 8: Unsupervised and Supervised Classification
**Objective**: Apply unsupervised and supervised classification techniques to remote sensing imagery for land cover mapping.

- **Key Features**:
  - Unsupervised classification methods
  - Supervised classification techniques
  - Land cover classification and mapping
- **Status**: This homework was completed during the [2026 Internet blackout in Iran](https://en.wikipedia.org/wiki/2026_Internet_blackout_in_Iran) and [2026 Iran massacres](https://en.wikipedia.org/wiki/2026_Iran_massacres). Due to these circumstances, no code is available for this assignment.

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

Each homework has its own directory with detailed README documentation. Here are quick start examples:

### Homework 1: Data Download
- Manual download from USGS EarthExplorer
- No scripts required (data acquisition only)

### Homework 2: ROI Selection and Color Composites
```bash
cd homework_02/scripts
python select_roi.py              # Plot image with ROI outline
python plot_color_composites.py   # Generate color composites
```

### Homework 3: LST Analysis
```bash
cd homework_03/scripts
python load_landsat5_tm_band6.py      # Select scene
python calculate_lst.py                # Full LST analysis
python calculate_lst_celsius.py        # Celsius map with histogram
python calculate_lst_rasht.py          # Rasht city analysis
```

### Homework 4: NDVI-LST Correlation
```bash
cd homework_04/scripts
python calculate_ndvi.py                    # Calculate NDVI
python create_vegetation_mask.py             # Create vegetation mask
python calculate_ndvi_lst_correlation.py     # Correlation analysis
python visualize_ndvi_lst_comparison.py      # Visualizations
```

### Homework 5: Geometric Correction
```bash
cd homework_05
python geometric_correction.py
```

### Homework 6: Scene Search and Registration
```bash
cd homework_06/scripts
python search_landsat_scenes.py              # Search scenes
python image_to_image_registration.py        # Image registration
python compare_slc_images.py                 # SLC comparison
```

### Homework 7: Change Detection
```bash
cd homework_07/scene_selection
python search_landsat_scenes.py              # Search scenes
python select_best_pair.py                   # Select optimal pair

cd ../scripts
python download_multispectral_bands.py      # Download data
python convert_to_reflectance.py             # Convert to reflectance
python calculate_reflectance_difference.py   # Calculate difference
python visualize_aoi_comparison.py           # Change detection viz
python visualize_diffusion_comparison.py     # Diffusion comparison
```

### Homework 8: Unsupervised and Supervised Classification
**Note**: No code is available for this homework. See [homework_08/README.md](homework_08/README.md) for details.

> **Note**: See individual homework README files for detailed usage instructions and configuration options.

## Requirements

See `requirements.txt` for full list. Main dependencies:
- `earthengine-api` - Google Earth Engine Python API
- `numpy` - Numerical computing
- `scipy` - Scientific computing and statistics
- `matplotlib` - Plotting and visualization
- `rasterio` - Geospatial raster I/O and processing
- `Pillow` - Image processing
- `jupyter` - Jupyter notebook support

## Output

All generated visualizations and analysis outputs are saved in each homework's `output/` directory.

## License

Academic use - Course assignment repository

## Course Completion

This repository documents the complete set of assignments for the Remote Sensing course (Fall 2025), covering fundamental topics from data acquisition and preprocessing to advanced analysis techniques including temperature analysis, change detection, and classification methods.
