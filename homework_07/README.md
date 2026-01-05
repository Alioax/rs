# Homework 7: Change Detection Analysis and Diffusion Comparison

## Overview

This homework assignment performs change detection analysis by comparing Landsat satellite imagery from two time periods (2015 and 2021) to identify reflectance changes. The analysis includes scene selection, data processing, difference calculation, and comprehensive visualization. Additionally, it includes a comparison of before/after diffusion processing to evaluate spatial resolution enhancement.

## Objectives

1. Search Landsat Collection 2 Level 2 archive for suitable scene pairs
2. Select optimal temporal pairs based on quality and phenological consistency
3. Download multispectral bands (B2-B7) from Google Earth Engine
4. Convert digital numbers to surface reflectance (0-1 range)
5. Calculate mean reflectance across all bands
6. Compute pixel-wise reflectance difference between time periods
7. Visualize change detection with AOI-based comparisons
8. Compare before/after diffusion processing (30m vs 15m resolution)

## Data Source

### Change Detection Images

- **Location**: Path 166, Row 34 (Northern Iran)
- **Satellites**: Landsat 8 OLI
- **Collection**: LANDSAT/LC08/C02/T1_L2 (Collection 2, Level 2 Surface Reflectance)
- **Selected Pair**:
  - **2015**: LC81660342015231LGN01 (August 19, 2015)
  - **2021**: LC81660342021231LGN00 (August 19, 2021)
- **Temporal Separation**: 6 years
- **Day of Year**: 231 (same day of year for phenological consistency)
- **Cloud Cover**: 0.66% (2015), 0.03% (2021)
- **Bands**: SR_B2 through SR_B7 (multispectral bands, excluding aerosol band)

### Diffusion Images

- **Before Diffusion**: Raw 30m Landsat bands 2-7
- **After Diffusion**: Diffused using 15m panchromatic band, resulting in 15m resolution bands 2-7

## Files

### Scene Selection Scripts
- `scene_selection/search_landsat_scenes.py` - Search GEE for suitable scenes (July/August, Path 166/Row 34)
- `scene_selection/select_best_pair.py` - Select optimal temporal pair based on quality and day-of-year matching

### Data Processing Scripts
- `scripts/download_multispectral_bands.py` - Export multispectral bands from GEE to Google Drive
- `scripts/convert_to_reflectance.py` - Convert DN values to surface reflectance (0-1)
- `scripts/calculate_mean_reflectance.py` - Calculate mean reflectance across all 6 bands
- `scripts/calculate_reflectance_difference.py` - Compute pixel-wise difference (2021 - 2015)

### Visualization Scripts
- `scripts/visualize_aoi_comparison.py` - Create change detection visualizations (3×3 grid, overview maps)
- `scripts/visualize_diffusion_comparison.py` - Create diffusion comparison visualizations (2×N grid, overview)

### Supporting Scripts
- `scripts/check_pixel_values.py` - Verify pixel value ranges
- `scripts/verify_reflectance.py` - Validate reflectance conversion

### Data Files
- `data/scenes_metadata.json` - All suitable scenes found in search (generated)
- `data/selected_pair.json` - Selected optimal pair with metadata (generated)
- `data/change_detection_images/` - Processed GeoTIFF files:
  - `*_multispectral_B2-7.tif` - Raw multispectral bands
  - `*_multispectral_B2-7_reflectance.tif` - Converted to reflectance
  - `*_mean_reflectance.tif` - Mean reflectance across bands
  - `reflectance_difference_2021-2015.tif` - Difference image
- `data/diffusion/` - Diffusion comparison images:
  - `Before Diffusion.tif` - Raw 30m resolution
  - `Diffusion.tif` - Diffused 15m resolution

### Output Files
- `output/aoi_overview.png` - Full 2021 RGB image with AOI outlines
- `output/aoi_overview_difference.png` - Reflectance difference map with AOI outlines
- `output/aoi_comparison_3x3.png` - 3×3 grid: 3 AOIs × 3 columns (2015 RGB, 2021 RGB, Difference)
- `output/diffusion_aoi_overview.png` - Before diffusion RGB with AOI outlines
- `output/diffusion_comparison_2xN.png` - 2×N grid: 2 rows (Before/After) × N columns (AOIs)

## Usage

### Complete Workflow

The analysis follows a sequential pipeline:

#### Step 1: Scene Search and Selection
```bash
cd homework_07/scene_selection
python search_landsat_scenes.py      # Search for suitable scenes
python select_best_pair.py            # Select optimal pair
```

#### Step 2: Data Download
```bash
cd ../scripts
python download_multispectral_bands.py  # Export to Google Drive (download manually)
```

**Note**: Files are exported to Google Drive in the 'EarthEngine' folder. Download them manually and place in `data/change_detection_images/`.

#### Step 3: Data Processing
```bash
cd scripts
python convert_to_reflectance.py           # Convert DN to reflectance
python calculate_mean_reflectance.py       # Calculate mean reflectance
python calculate_reflectance_difference.py # Compute difference
```

#### Step 4: Visualization
```bash
python visualize_aoi_comparison.py      # Change detection visualizations
python visualize_diffusion_comparison.py # Diffusion comparison visualizations
```

## Methodology

### 1. Scene Selection

The selection process prioritizes:
- **Temporal Consistency**: Minimizes day-of-year difference (same day of year preferred)
- **Quality**: Filters by cloud cover (≤20%), cloud shadow (≤10%), snow (≤5%)
- **Satellite Compatibility**: Uses same satellite (Landsat 8 OLI) for consistency
- **Year Separation**: Ensures sufficient temporal gap (≥1 year) for change detection

**Scoring Formula**: `score = (day_of_year_difference × 10) + combined_cloud_cover`

Lower scores indicate better pairs.

### 2. Surface Reflectance Conversion

Landsat Collection 2 Level 2 data is converted from Digital Numbers (DN) to surface reflectance:

**Formula**: `Reflectance = (DN × 0.0000275) - 0.2`

- Values are clipped to valid reflectance range (0-1)
- Output data type: float32
- Preserves georeferencing information

### 3. Mean Reflectance Calculation

Mean reflectance is calculated across all 6 multispectral bands (B2-B7) for each pixel:

**Formula**: `Mean_Reflectance = mean(B2, B3, B4, B5, B6, B7)`

This provides a single-band representation of overall surface reflectance.

### 4. Reflectance Difference Calculation

Pixel-wise difference is calculated between the two time periods:

**Formula**: `Difference = Reflectance_2021 - Reflectance_2015`

- **Positive values**: Increase in reflectance (e.g., urbanization, deforestation)
- **Negative values**: Decrease in reflectance (e.g., vegetation growth, water increase)
- **Zero values**: No change

**Spatial Alignment**:
- If images are perfectly aligned: Direct subtraction
- If misaligned: Reprojects older image to match newer image grid using bilinear resampling

### 5. Change Detection Visualization

#### AOI Configuration
Three Areas of Interest (AOIs) are defined for detailed comparison:
- **AOI 1**: Center (5400, 4100), 400×400 pixels
- **AOI 2**: Center (4000, 4200), 400×400 pixels
- **AOI 3**: Center (3650, 6700), 400×400 pixels

#### Visualization Products
1. **Overview Maps**: Full scene with AOI outlines
2. **3×3 Comparison Grid**: 
   - Rows: 3 AOIs
   - Columns: 2015 RGB, 2021 RGB, Difference map
   - Uses diverging colormap (RdBu_r) for difference visualization

#### Normalization
- RGB composites: Percentile-based stretching (2-98 percentile) per channel
- Difference map: Symmetric normalization around zero for diverging colormap
- Consistent scaling per column for fair comparison

### 6. Diffusion Comparison

Compares before/after diffusion processing to evaluate resolution enhancement:

- **Before**: 30m resolution (raw Landsat multispectral bands)
- **After**: 15m resolution (diffused using panchromatic band)

**Spatial Alignment**:
- After image (15m) used as reference grid to preserve higher resolution
- Before image (30m) reprojected to match after image's 15m grid
- AOI coordinates converted from 30m to 15m pixel space

**Visualization**:
- **2×N Grid**: 2 rows (Before/After) × N columns (AOIs)
- Overview uses before image (30m) for reference
- Grid comparison shows both at 15m resolution for fair comparison

## Expected Results

### Change Detection

The analysis should reveal:
- **Areas of increase**: Urban expansion, deforestation, land clearing
- **Areas of decrease**: Vegetation growth, water body expansion, agricultural development
- **Stable areas**: No significant change over 6-year period

### Statistical Summary

The difference calculation provides:
- Minimum and maximum difference values
- Mean difference (overall trend)
- Standard deviation (variability)
- Percentage of pixels with increase, decrease, or no change

### Visual Patterns

- **RGB Composites**: Natural color visualization (B4, B3, B2)
- **Difference Map**: 
  - Red tones: Increase in reflectance
  - Blue tones: Decrease in reflectance
  - White/neutral: No change

### Diffusion Comparison

- **Resolution Enhancement**: After diffusion should show finer detail at 15m resolution
- **Spatial Alignment**: Both images properly aligned for pixel-by-pixel comparison
- **Visual Quality**: Higher resolution image should reveal more spatial detail

## Output Files

### Change Detection Outputs

1. **aoi_overview.png**
   - Full 2021 RGB image (7638 × 7877 pixels)
   - AOI outlines with colored borders (red, lime, cyan)
   - Geographic reference

2. **aoi_overview_difference.png**
   - Full reflectance difference map
   - Diverging colormap (RdBu_r)
   - AOI outlines
   - Colorbar with reflectance difference values

3. **aoi_comparison_3x3.png**
   - 3×3 grid visualization
   - Row 1-3: AOI 1, AOI 2, AOI 3
   - Column 1: 2015 RGB
   - Column 2: 2021 RGB
   - Column 3: Difference map
   - Horizontal colorbar above grid

### Diffusion Comparison Outputs

1. **diffusion_aoi_overview.png**
   - Full before diffusion RGB image (30m)
   - AOI outlines with colored borders

2. **diffusion_comparison_2xN.png**
   - 2×N grid visualization (N = number of AOIs)
   - Row 1: Before Diffusion
   - Row 2: After Diffusion
   - Columns: One per AOI
   - Both images at 15m resolution for comparison

## Notes

### Data Processing

- Scripts use relative paths and should be run from appropriate directories
- Google Earth Engine authentication required: Run `ee.Authenticate()` on first use
- Exported files from GEE must be downloaded manually from Google Drive
- All processing preserves georeferencing information

### Spatial Alignment

- Images are checked for spatial alignment (CRS, transform, shape)
- Automatic reprojection/resampling if needed
- Change detection: Newer image used as reference
- Diffusion comparison: After image (15m) used as reference

### AOI Configuration

- AOI coordinates are configurable in visualization scripts
- Change detection and diffusion comparison use independent AOI configurations
- Coordinates are in pixel space of reference image

### Colorbar Values

- Colorbar labels show "Reflectance Difference (2021 - 2015)"
- Values formatted to 3 decimal places
- Symmetric range around zero for diverging colormap
- Consistent normalization between overview and grid visualizations

## Dependencies

Required:
- `numpy` - Numerical operations
- `matplotlib` - Visualization
- `rasterio` - Geospatial raster I/O and processing
- `earthengine-api` - Google Earth Engine Python API
- `tqdm` - Progress bars
- `json` - JSON file handling

Optional:
- `fiona`, `shapely`, `pyproj` - For advanced geospatial operations (if needed)

See `requirements.txt` in the repository root for full list.

## Key Takeaways

1. **Temporal Consistency**: Matching day of year minimizes phenological differences
2. **Quality Filtering**: Cloud cover and quality metrics ensure usable imagery
3. **Surface Reflectance**: Level 2 data provides atmospherically corrected reflectance
4. **Spatial Alignment**: Proper alignment critical for accurate change detection
5. **Visualization**: AOI-based approach enables detailed local analysis
6. **Resolution Enhancement**: Diffusion processing can improve spatial resolution from 30m to 15m

## References

- Landsat Collection 2 Level 2 Surface Reflectance Data Format
- Google Earth Engine API Documentation
- Change Detection in Remote Sensing
- Pan-sharpening and Resolution Enhancement Techniques
- Surface Reflectance vs. Top-of-Atmosphere Reflectance

