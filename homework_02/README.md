# Homework 2: Selecting AOI and Color Composites

## Overview

This homework assignment focuses on selecting a Region of Interest (ROI) from a multispectral Landsat 8 image and visualizing it using different color composite combinations. The analysis includes interactive ROI selection and comparison of various band combinations for remote sensing interpretation.

## Objectives

1. Load multispectral Landsat 8 image (bands 2-7) from GeoTIFF file
2. Create a default RGB composite for visualization
3. Define and visualize a configurable rectangular ROI on the full image
4. Extract the ROI region from all bands
5. Generate multiple color composites based on configuration list
6. Compare different band combinations for various remote sensing applications

## Data Source

- **Satellite**: Landsat 8 OLI
- **Bands**: B2, B3, B4, B5, B6, B7 (6 multispectral bands)
- **File**: `data/Image.tif` - GeoTIFF format with 6 bands
- **Band Order**: B2 (index 0), B3 (index 1), B4 (index 2), B5 (index 3), B6 (index 4), B7 (index 5)

## Files

### Main Scripts

- `scripts/select_roi.py` - Plot full image with configurable rectangular ROI outline
- `scripts/plot_color_composites.py` - Extract ROI and plot multiple color composites

### Data

- `data/Image.tif` - Input multispectral image (bands 2-7)

### Outputs

- `output/roi_selection.png` - Full image with ROI outline (from select_roi.py)
- `output/color_composites_roi.png` - Grid of color composites for ROI (from plot_color_composites.py)

## Usage

### Step 1: Select ROI

First, run the ROI selection script to visualize the image and define your ROI:

```bash
cd homework_02/scripts
python select_roi.py
```

This will:
1. Load the multispectral image
2. Create a default RGB composite (432: B4, B3, B2)
3. Plot the full image with a rectangular ROI outline
4. Save the visualization to `output/roi_selection.png`

**Configure ROI**: Edit the `roi_config` dictionary in `select_roi.py`:
```python
roi_config = {
    'center': (2000, 1500),  # (col, row) in pixels
    'x_side_length': 500,    # width in pixels
    'y_side_length': 400,    # height in pixels
    'name': 'ROI 1'
}
```

### Step 2: Plot Color Composites

After selecting your ROI, run the color composite script:

```bash
python plot_color_composites.py
```

This will:
1. Load the multispectral image
2. Extract the ROI region from all bands
3. Create color composites based on the configuration list
4. Plot all composites in a grid layout
5. Save the visualization to `output/color_composites_roi.png`

**Configure Color Composites**: Edit the `color_composites` list in `plot_color_composites.py`:
```python
color_composites = ['432', '543', '436']  # List of band combinations
```

**Important**: The `roi_config` in `plot_color_composites.py` must match the one in `select_roi.py` to extract the same ROI.

## ROI Configuration

The ROI is defined as a rectangular region with:

- **center**: `(col, row)` tuple in pixel coordinates
  - `col`: Column (x) coordinate
  - `row`: Row (y) coordinate
- **x_side_length**: Width of the rectangle in pixels
- **y_side_length**: Height of the rectangle in pixels
- **name**: Descriptive name for the ROI

The ROI is centered at the specified coordinates and extends `x_side_length/2` pixels in each horizontal direction and `y_side_length/2` pixels in each vertical direction.

## Color Composite Configuration

Color composites are specified as 3-digit strings where each digit represents a Landsat band number:

- **Band Numbers**: B2, B3, B4, B5, B6, B7
- **Format**: `'RGB'` where R, G, B are band numbers (2-7)
- **Example**: `'432'` means:
  - Red channel: B4 (index 2)
  - Green channel: B3 (index 1)
  - Blue channel: B2 (index 0)

### Common Color Composite Examples

- **432**: Natural color (B4=R, B3=G, B2=B) - True-color visualization
- **543**: False color (B5=R, B4=G, B3=B) - Good for vegetation analysis
- **436**: False color (B4=R, B3=G, B6=B) - SWIR for moisture/vegetation
- **754**: False color (B7=R, B5=G, B4=B) - SWIR combinations for burn scars
- **564**: False color (B5=R, B6=G, B4=B) - Vegetation and moisture
- **765**: False color (B7=R, B6=G, B5=B) - Advanced SWIR analysis

## Methodology

### 1. Image Loading

The multispectral image is loaded using rasterio:
- Reads all 6 bands (B2-B7) as a 3D array
- Shape: `[bands, height, width]`
- Preserves georeferencing information

### 2. RGB Composite Creation

RGB composites are created by:
1. Selecting three bands based on composite code
2. Mapping Landsat band numbers to array indices:
   - B2 → index 0
   - B3 → index 1
   - B4 → index 2
   - B5 → index 3
   - B6 → index 4
   - B7 → index 5
3. Stacking bands to create RGB array: `[height, width, 3]`

### 3. Normalization

RGB data is normalized for display using percentile-based stretching:
- **Method**: 2-98 percentile clipping per channel (1-99 for color composites)
- **Filtering**: Only positive pixel values are used (excludes zeros and negatives)
- **Output**: uint8 (0-255) for display
- **Purpose**: Enhances contrast while avoiding outlier effects and reducing hazy appearance

### 4. ROI Extraction

Rectangular ROI extraction:
1. Calculate bounding box from center and side lengths
2. Clamp to image bounds to handle edge cases
3. Extract region from all bands
4. Return ROI bands array: `[bands, roi_height, roi_width]`

### 5. Visualization

- **ROI Selection**: 
  - Full image with rectangular outline using matplotlib.patches.Rectangle
  - "Region of Interest" annotation at the bottom of the ROI
  - Clean visualization without axes, grid, labels, or title
- **Color Composites**: Grid layout with automatic sizing based on number of composites
- **High Resolution**: 300 DPI output for publication quality

## Output Files

### roi_selection.png

- Full multispectral image displayed as RGB composite (432)
- Rectangular ROI outline in red
- "Region of Interest" annotation at the bottom center of the ROI
- Clean visualization without axes, grid, or labels for publication-ready output

### color_composites_roi.png

- Grid of color composite visualizations
- Each composite labeled with band combination
- Consistent normalization across all composites
- ROI information in title

## Configuration Examples

### Example 1: Small Urban Area

```python
roi_config = {
    'center': (1500, 1200),
    'x_side_length': 300,
    'y_side_length': 300,
    'name': 'Urban Area'
}

color_composites = ['432', '543', '754']  # Natural, vegetation, burn
```

### Example 2: Agricultural Field

```python
roi_config = {
    'center': (2500, 1800),
    'x_side_length': 600,
    'y_side_length': 500,
    'name': 'Agricultural Field'
}

color_composites = ['432', '543', '564']  # Natural, vegetation, moisture
```

### Example 3: Water Body

```python
roi_config = {
    'center': (1000, 2000),
    'x_side_length': 400,
    'y_side_length': 400,
    'name': 'Water Body'
}

color_composites = ['432', '436', '543']  # Natural, SWIR, vegetation
```

## Notes

- Scripts use relative paths and should be run from the `scripts/` directory
- Outputs are saved to `output/` directory (created automatically)
- ROI configuration must be consistent between both scripts
- Color composite codes must use valid Landsat band numbers (2-7)
- The scripts handle edge cases where ROI extends beyond image bounds
- All visualizations use high-resolution settings (300 DPI)

## Dependencies

Required packages (already in `requirements.txt`):
- `numpy` - Numerical operations
- `matplotlib` - Visualization
- `rasterio` - Geospatial raster I/O
- `pathlib` - Path handling

## Troubleshooting

### ROI extends beyond image bounds
- The scripts automatically clamp ROI bounds to image dimensions
- Check that center coordinates are within image size

### Invalid composite code error
- Ensure composite codes are 3 digits
- Band numbers must be between 2 and 7
- Example valid codes: '432', '543', '436', '754'

### Image file not found
- Ensure `Image.tif` exists in `homework_02/data/` directory
- Check file path and permissions

## Key Takeaways

1. **ROI Selection**: Rectangular ROIs allow flexible area selection for analysis
2. **Color Composites**: Different band combinations reveal different features:
   - Natural color (432): True-color visualization
   - False color (543): Enhanced vegetation detection
   - SWIR combinations: Moisture, burn scars, geological features
3. **Normalization**: Percentile-based stretching with zero/negative filtering improves visualization quality and reduces hazy appearance
4. **Band Mapping**: Understanding Landsat band numbers vs. array indices is crucial

## References

- Landsat 8 OLI Band Specifications
- Remote Sensing Color Composite Techniques
- Percentile-based Image Stretching Methods

