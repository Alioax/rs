# Homework 4: Relationship Between Vegetation Cover and Land Surface Temperature

## Overview

This homework assignment quantitatively analyzes the relationship between vegetation cover (NDVI) and land surface temperature (LST) using satellite remote sensing data. It builds directly on Homework 3 by reusing the same Landsat 5 TM satellite image and the previously derived LST layer, ensuring consistency and comparability.

## Objectives

1. Load the exact same Landsat 5 TM image used in Homework 3
2. Convert reflective bands (Red and NIR) to Top-of-Atmosphere (TOA) reflectance
3. Calculate Normalized Difference Vegetation Index (NDVI)
4. Recalculate Land Surface Temperature (LST) using Homework 3 methodology
5. Create vegetation mask (NDVI > 0.35)
6. Apply mask to restrict analysis to vegetated areas
7. Perform correlation analysis between NDVI and LST
8. Visualize results with maps and scatter plots

## Data Source

- **Satellite**: Landsat 5 Thematic Mapper (TM)
- **Collection**: LANDSAT/LT05/C02/T1 (Collection 2, Tier 1)
- **Scene**: LT51660341987186XXX02 (July 5, 1987)
- **Location**: Path 166, Row 34 (Northern Iran)
- **Cloud Cover**: 0%

## Files

### Main Notebook
- `homework_04_ndvi_lst_analysis.ipynb` - Complete analysis notebook with all steps

### Supporting Scripts
- `scripts/calculate_ndvi.py` - Convert bands to reflectance and calculate NDVI
- `scripts/create_vegetation_mask.py` - Generate binary vegetation mask (NDVI > 0.35)
- `scripts/calculate_ndvi_lst_correlation.py` - Compute correlation between masked NDVI and LST
- `scripts/visualize_ndvi_lst_comparison.py` - Multi-panel visualizations and scatter plots

### Data
- `data/selected_scene.json` - Scene metadata (reused from Homework 3)
- `data/correlation_results.json` - Statistical correlation results (generated)

### Outputs
- `output/NDVI_[scene_id].png` - Full NDVI map
- `output/VegetationMask_[scene_id].png` - Binary vegetation mask
- `output/NDVI_Masked_[scene_id].png` - NDVI for vegetated areas only
- `output/LST_Masked_[scene_id].png` - LST for vegetated areas only
- `output/NDVI_LST_Scatter_[scene_id].png` - Scatter plot with correlation
- `output/NDVI_LST_Comparison_[scene_id].png` - Multi-panel comparison

## Usage

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook homework_04_ndvi_lst_analysis.ipynb
```

### Option 2: Python Scripts
Run scripts in order:
```bash
cd scripts
python calculate_ndvi.py                    # Step 1: Calculate NDVI
python create_vegetation_mask.py             # Step 2: Create vegetation mask
python calculate_ndvi_lst_correlation.py     # Step 3: Correlation analysis
python visualize_ndvi_lst_comparison.py      # Step 4: Multi-panel visualization
```

## Methodology

### 1. TOA Reflectance Conversion

For Landsat 5 TM Collection 2, bands are converted from Digital Numbers (DN) to Top-of-Atmosphere (TOA) reflectance using metadata:

**Formula**: ρ = REFLECTANCE_MULT_BAND_X × DN + REFLECTANCE_ADD_BAND_X

Where:
- ρ = TOA reflectance (0-1 range)
- REFLECTANCE_MULT_BAND_X = Multiplicative scaling factor
- REFLECTANCE_ADD_BAND_X = Additive scaling factor

### 2. NDVI Calculation

Normalized Difference Vegetation Index is calculated from reflectance values:

**Formula**: NDVI = (NIR_reflectance - Red_reflectance) / (NIR_reflectance + Red_reflectance)

- NDVI values range from -1 to +1
- Higher values indicate denser vegetation
- Typical ranges:
  - < 0: Water, clouds, snow
  - 0 - 0.2: Bare soil, urban areas
  - 0.2 - 0.4: Sparse vegetation
  - 0.4 - 0.6: Moderate vegetation
  - > 0.6: Dense vegetation

### 3. LST Calculation

Land Surface Temperature is calculated using the same methodology from Homework 3:

1. **DN to Radiance**: L = RADIANCE_MULT_BAND_6 × B6 + RADIANCE_ADD_BAND_6
2. **Radiance to Kelvin**: T_K = K2 / ln((K1 × ε / L) + 1) where ε = 0.95
3. **Kelvin to Celsius**: T_C = T_K - 273.15
4. **Resample to 30m**: Match NDVI resolution for spatial consistency

### 4. Vegetation Masking

A binary mask is created to identify vegetated areas:

**Threshold**: NDVI > 0.35

- **Rationale**: This threshold effectively separates vegetated areas from bare soil, water, and urban areas
- **Mask values**: 1 = Vegetated, 0 = Non-vegetated
- The mask is applied to both NDVI and LST layers to restrict correlation analysis to vegetated pixels only

### 5. Correlation Analysis

Pearson correlation coefficient is calculated between masked NDVI and LST values:

- **Sampling**: Systematic sampling at 30m resolution (target: 10,000 pixels)
- **Method**: Pearson correlation coefficient (r)
- **Interpretation**:
  - |r| < 0.3: Weak correlation
  - 0.3 ≤ |r| < 0.7: Moderate correlation
  - |r| ≥ 0.7: Strong correlation
  - Negative r: Higher NDVI → Lower LST (cooling effect)
  - Positive r: Higher NDVI → Higher LST (unusual, may indicate other factors)

## Expected Results

### Correlation Interpretation

A **strong negative correlation** indicates that:
- Higher vegetation density (NDVI) corresponds to lower surface temperatures (LST)
- Vegetation has a cooling effect on land surface temperature
- This is the expected relationship due to:
  - **Shading**: Vegetation canopy shades the ground surface
  - **Evapotranspiration**: Plants release water vapor, which cools the surface
  - **Albedo**: Vegetation typically has higher albedo than bare soil

### Spatial Patterns

The analysis should reveal:
- **Vegetated areas**: Lower temperatures, higher NDVI values
- **Bare soil/urban areas**: Higher temperatures, lower NDVI values
- **Water bodies**: May have low NDVI but moderate temperatures

## Output Files

### Maps
- **NDVI Map**: Shows vegetation density across the scene
- **LST Map**: Shows temperature distribution
- **Masked Maps**: Focus on vegetated areas only

### Statistical Results
- **Correlation coefficient**: Quantitative measure of NDVI-LST relationship
- **P-value**: Statistical significance
- **Sample size**: Number of pixels analyzed
- **Interpretation**: Strength and direction of correlation

### Visualizations
- **Scatter plot**: NDVI vs LST with trend line
- **Multi-panel comparison**: Side-by-side maps and scatter plot

## Notes

- Scripts use relative paths and should be run from the `scripts/` directory
- Outputs are saved to `output/` directory
- Scene metadata is reused from Homework 3
- All analysis is performed at 30m resolution for spatial consistency
- Correlation analysis focuses on vegetated areas only (NDVI > 0.35)

## Dependencies

Same as Homework 3, plus:
- `scipy` - For statistical correlation calculations

See `requirements.txt` in the repository root for full list.

## References

- Landsat 5 TM Collection 2 Data Format Control Book (DFCB)
- NDVI: Normalized Difference Vegetation Index
- LST: Land Surface Temperature calculation methodology from Homework 3

