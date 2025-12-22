# Homework 5: Geometric Correction and Resampling Method Comparison

## Overview

This homework assignment performs geometric correction of satellite images using ground control points (GCPs) and evaluates the effects of different resampling methods on image quality. The analysis focuses on positional accuracy assessment, error evaluation, and visual comparison of resampling techniques.

## Objectives

1. Load unrectified satellite images (B2, B3, B4 bands)
2. Parse ground control points (GCPs) from ENVI format file
3. Perform geometric correction using GCPs
4. Calculate Root Mean Square Error (RMSE) for accuracy assessment
5. Apply three resampling methods (Nearest Neighbor, Bilinear, Cubic Convolution)
6. Overlay road network for visual verification
7. Compare resampling methods across different Areas of Interest (AOIs)
8. Generate comprehensive visualizations and reports

## Data Source

- **Location**: Shiraz, Iran
- **Coordinate System**: UTM Zone 39N (EPSG:32639), WGS84
- **Bands**: B2 (Green), B3 (Red), B4 (Near Infrared)
- **Reference Data**: Road network shapefile
- **GCPs**: 14 ground control points in ENVI format

## Files

### Main Script
- `geometric_correction.py` - Complete geometric correction and resampling comparison script

### Input Data
- `Shiraz_Georeferencing/Unrectified/B2.TIF` - Unrectified green band
- `Shiraz_Georeferencing/Unrectified/B3.TIF` - Unrectified red band
- `Shiraz_Georeferencing/Unrectified/B4.TIF` - Unrectified near-infrared band
- `Shiraz_Georeferencing/Road/road-line.shp` - Road network shapefile
- `points.pts` - Ground control points file (ENVI format)

### Outputs
- `output/corrected_B2.tif` - Geometrically corrected green band
- `output/corrected_B3.tif` - Geometrically corrected red band
- `output/corrected_B4.tif` - Geometrically corrected near-infrared band
- `output/resampled_B4_nearest.tif` - Band 4 resampled with Nearest Neighbor
- `output/resampled_B4_bilinear.tif` - Band 4 resampled with Bilinear interpolation
- `output/resampled_B4_cubic.tif` - Band 4 resampled with Cubic Convolution
- `output/resampled_B4_stacked.tif` - Stacked GeoTIFF with all three resampling methods
- `output/rectified_with_roads.png` - Rectified image with road overlay and GCPs
- `output/resampling_comparison_3x3.png` - 3×3 grid comparing resampling methods across AOIs
- `output/aoi_overview.png` - Overview image showing AOI locations
- `output/gcp_rmse_report.txt` - RMSE report with GCP table and individual errors

## Usage

Run the main script:
```bash
cd homework_05
python geometric_correction.py
```

The script will:
1. Load unrectified images
2. Parse GCPs from `points.pts`
3. Perform geometric correction for all bands
4. Calculate RMSE
5. Apply resampling methods to Band 4
6. Create visualizations with road overlay
7. Generate AOI comparison grids
8. Save all outputs to `output/` directory

## Methodology

### 1. Ground Control Point (GCP) Parsing

GCPs are parsed from ENVI format file (`points.pts`):
- Format: Map X, Map Y, Image X, Image Y
- 14 GCPs distributed across the image
- Target CRS: UTM Zone 39N (EPSG:32639)

### 2. Geometric Correction

Geometric correction is performed using rasterio's GCP-based transformation:

1. **GCP Creation**: Convert parsed coordinates to rasterio `GroundControlPoint` objects
2. **Transform Calculation**: Use `calculate_default_transform()` to compute transformation matrix
3. **Reprojection**: Apply transformation using `reproject()` with GCPs
4. **Output**: Generate georeferenced GeoTIFF files with proper CRS and transform

**Key Parameters**:
- Source CRS: None (unrectified image)
- Destination CRS: EPSG:32639 (UTM Zone 39N)
- Pixel size: ~30 meters (calculated from transform)

### 3. RMSE Calculation

Root Mean Square Error is calculated to assess correction accuracy:

**Formula**: 
- RMSE_X = √(Σ(error_x²) / n)
- RMSE_Y = √(Σ(error_y²) / n)
- Total RMSE = √(Σ(total_error²) / n)

Where:
- error_x = predicted_x - actual_x (easting error)
- error_y = predicted_y - actual_y (northing error)
- total_error = √(error_x² + error_y²)

**Interpretation**:
- RMSE < 1 pixel: Excellent accuracy
- RMSE < 2 pixels: Good accuracy
- RMSE > 2 pixels: May need GCP refinement

### 4. Resampling Methods

Three resampling methods are applied to Band 4 for comparison:

#### Nearest Neighbor
- **Method**: Assigns value of nearest pixel
- **Advantages**: Preserves original pixel values, no interpolation
- **Use Case**: Quantitative analysis, classification
- **Disadvantages**: Can create blocky appearance

#### Bilinear Interpolation
- **Method**: Weighted average of 4 nearest pixels
- **Advantages**: Smooth appearance, reduces blockiness
- **Use Case**: Visual interpretation, smooth surfaces
- **Disadvantages**: Modifies pixel values, may blur edges

#### Cubic Convolution
- **Method**: Weighted average using cubic function (16 pixels)
- **Advantages**: Smoothest appearance, preserves edges better than bilinear
- **Use Case**: High-quality visualizations
- **Disadvantages**: Most computationally intensive, modifies pixel values significantly

### 5. Road Overlay Verification

Road network overlay is used to visually verify geometric correction accuracy:
- Roads are loaded from shapefile
- Reprojected if CRS differs from image CRS
- Overlaid as thick yellow lines (3.0 px width, fully opaque)
- Visual inspection confirms proper alignment

### 6. Areas of Interest (AOIs)

Three AOIs are defined for resampling comparison:
- **AOI 1**: Center (row=100, col=150), 50×50 pixels
- **AOI 2**: Center (row=210, col=298), 50×50 pixels
- **AOI 3**: Center (row=250, col=175), 50×50 pixels

Each AOI represents different spatial characteristics:
- Dense urban areas
- Curved linear features
- Peripheral/low-contrast areas

## Expected Results

### Geometric Correction

- **Input**: 300 × 400 pixels (unrectified)
- **Output**: ~354 × 439 pixels (rectified, depends on transformation)
- **Pixel Size**: ~30.02 × 30.02 meters
- **RMSE**: Should be < 1 pixel for acceptable accuracy

### Resampling Comparison

The 3×3 comparison grid should reveal:
- **Nearest Neighbor**: Sharp, blocky appearance, preserves original values
- **Bilinear**: Smooth appearance, slight blurring of edges
- **Cubic Convolution**: Smoothest appearance, best edge preservation

### Visual Verification

- Roads should align well with road features in the image
- GCPs should be located at recognizable features (road intersections, etc.)
- Rectified image should show proper georeferencing

## Output Files

### Corrected Images
- Three corrected bands (B2, B3, B4) as GeoTIFF files
- Proper georeferencing with UTM Zone 39N CRS
- Can be used for further analysis or GIS applications

### Resampled Images
- Individual resampled Band 4 files for each method
- Stacked GeoTIFF with all three methods (3 bands)

### Visualizations
- **Rectified with Roads**: Shows corrected image with road overlay and GCPs
- **3×3 Comparison Grid**: Side-by-side comparison of resampling methods
- **AOI Overview**: Full image with AOI locations highlighted

### Reports
- **RMSE Report**: Detailed GCP table with individual errors and summary statistics

## Notes

- Script uses relative paths and should be run from `homework_05/` directory
- All outputs are saved to `output/` directory
- Road overlay requires `fiona`, `shapely`, and `pyproj` libraries
- If road libraries are unavailable, script will skip road overlay but continue processing
- GCPs are displayed as cyan circles with dark blue edges on the visualization
- All visualizations use high-resolution settings (300 DPI)

## Dependencies

Required:
- `numpy` - Numerical operations
- `matplotlib` - Visualization
- `rasterio` - Geospatial raster I/O and processing
- `pandas` - Data manipulation (for GCP table)

Optional (for road overlay):
- `fiona` - Shapefile reading
- `shapely` - Geometric operations
- `pyproj` - Coordinate transformations

See `requirements.txt` in the repository root for full list.

## Key Takeaways

1. **GCP Selection**: Proper GCP selection and spatial distribution are critical for low geometric error
2. **RMSE Assessment**: RMSE is a key metric for evaluating correction quality
3. **Resampling Choice**: Resampling method should match intended use:
   - **Nearest Neighbor**: For quantitative analysis (preserves pixel values)
   - **Bilinear/Cubic**: For visual interpretation (smoother appearance)
4. **Visual Verification**: Road overlay and GCP visualization help verify correction accuracy

## References

- ENVI Image to Map GCP File Format
- Rasterio Documentation: Geometric Correction with GCPs
- UTM Zone 39N (EPSG:32639) Coordinate System
- Resampling Methods in Remote Sensing

