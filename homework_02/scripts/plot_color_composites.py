"""
Color Composite Visualization for ROI

This script:
1. Loads multispectral Landsat 8 image (bands 2-7) from Image.tif
2. Extracts a configurable rectangular ROI from all bands
3. Creates multiple color composites based on configuration list
4. Plots all composites in a grid layout

Author: Ali Haghighi
Course: Remote Sensing
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib parameters for high-quality plots
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define paths
script_dir = Path(__file__).parent.absolute()
homework_dir = script_dir.parent
data_dir = homework_dir / 'data'
output_dir = homework_dir / 'output'
output_dir.mkdir(exist_ok=True, parents=True)

# Input file
image_file = data_dir / 'Image.tif'

# ROI Configuration (must match select_roi.py)
roi_config = {
    'center': (2500, 4000),  # (col, row) in pixels
    'x_side_length': 1600,    # width in pixels
    'y_side_length': 900,    # height in pixels
    'name': 'ROI 1'
}

# Color Composite Configuration
# List of band combinations as strings (e.g., '432' means B4=R, B3=G, B2=B)
# Band numbers: B2, B3, B4, B5, B6, B7
color_composites = ['432', '543', '436']  # Natural color, false color, SWIR false color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_multispectral_image(image_path):
    """
    Load multispectral image from GeoTIFF file.
    
    Parameters:
    -----------
    image_path : Path
        Path to the GeoTIFF file
    
    Returns:
    --------
    numpy.ndarray
        3D array of shape (bands, height, width) - bands are B2, B3, B4, B5, B6, B7
    dict
        Rasterio profile information
    """
    with rasterio.open(image_path) as src:
        bands = src.read()  # Shape: [bands, height, width]
        profile = src.profile.copy()
        
        print(f"  [OK] Loaded image: {src.width} × {src.height} pixels, {src.count} bands")
        
        return bands, profile


def parse_composite_code(code):
    """
    Parse color composite code string to band indices.
    
    Parameters:
    -----------
    code : str
        Color composite code (e.g., '432' means B4=R, B3=G, B2=B)
        Band numbers: B2=2, B3=3, B4=4, B5=5, B6=6, B7=7
        Array indices: B2=0, B3=1, B4=2, B5=3, B6=4, B7=5
    
    Returns:
    --------
    list
        [R_index, G_index, B_index] for array indexing
    str
        Description of the composite
    """
    if len(code) != 3:
        raise ValueError(f"Composite code must be 3 digits, got: {code}")
    
    # Parse band numbers
    r_band_num = int(code[0])
    g_band_num = int(code[1])
    b_band_num = int(code[2])
    
    # Validate band numbers (B2-B7)
    valid_bands = [2, 3, 4, 5, 6, 7]
    if r_band_num not in valid_bands or g_band_num not in valid_bands or b_band_num not in valid_bands:
        raise ValueError(f"Invalid band numbers. Must be 2-7, got: {code}")
    
    # Convert band numbers to array indices (B2=0, B3=1, B4=2, B5=3, B6=4, B7=5)
    r_index = r_band_num - 2
    g_index = g_band_num - 2
    b_index = b_band_num - 2
    
    # Create description
    description = f"RGB{r_band_num}{g_band_num}{b_band_num} (B{r_band_num}=R, B{g_band_num}=G, B{b_band_num}=B)"
    
    return [r_index, g_index, b_index], description


def extract_rectangular_roi(band_data, center_row, center_col, x_side_length, y_side_length):
    """
    Extract a rectangular region from a band array.
    
    Parameters:
    -----------
    band_data : numpy.ndarray
        2D array of band data
    center_row : int
        Row coordinate of center pixel
    center_col : int
        Column coordinate of center pixel
    x_side_length : int
        Width of rectangle in pixels
    y_side_length : int
        Height of rectangle in pixels
    
    Returns:
    --------
    numpy.ndarray
        Extracted rectangular region
    tuple
        (row_start, row_end, col_start, col_end) - actual bounds used
    """
    height, width = band_data.shape
    
    # Calculate half lengths
    half_x = x_side_length // 2
    half_y = y_side_length // 2
    
    # Calculate bounding box
    row_start = center_row - half_y
    row_end = center_row + half_y + (y_side_length % 2)
    col_start = center_col - half_x
    col_end = center_col + half_x + (x_side_length % 2)
    
    # Clamp to image bounds
    row_start = max(0, int(row_start))
    row_end = min(height, int(row_end))
    col_start = max(0, int(col_start))
    col_end = min(width, int(col_end))
    
    # Extract region
    region = band_data[row_start:row_end, col_start:col_end]
    
    return region, (row_start, row_end, col_start, col_end)


def create_rgb_composite(bands_data, band_indices=[2, 1, 0]):
    """
    Create RGB composite from multispectral bands.
    
    Parameters:
    -----------
    bands_data : numpy.ndarray
        3D array of shape (bands, height, width) - bands are B2, B3, B4, B5, B6, B7
    band_indices : list
        Indices for R, G, B channels [R_idx, G_idx, B_idx]
        Default [2, 1, 0] means B4 (R), B3 (G), B2 (B) for natural color
    
    Returns:
    --------
    numpy.ndarray
        RGB composite of shape (height, width, 3)
    """
    if bands_data.shape[0] < max(band_indices) + 1:
        raise ValueError(f"Not enough bands. Need at least {max(band_indices) + 1} bands")
    
    # Extract RGB bands
    r_band = bands_data[band_indices[0], :, :]  # Red channel
    g_band = bands_data[band_indices[1], :, :]  # Green channel
    b_band = bands_data[band_indices[2], :, :]  # Blue channel
    
    # Stack to create RGB
    rgb = np.dstack([r_band, g_band, b_band])
    
    return rgb


def normalize_rgb_for_display(rgb_data, percentile_low=1, percentile_high=99):
    """
    Normalize RGB data for display using percentile-based stretching.
    
    Parameters:
    -----------
    rgb_data : numpy.ndarray
        3D array of shape (height, width, 3) with RGB channels
    percentile_low : float
        Lower percentile for stretching
    percentile_high : float
        Upper percentile for stretching
    
    Returns:
    --------
    numpy.ndarray
        Normalized RGB data (uint8, 0-255)
    """
    normalized = np.zeros_like(rgb_data, dtype=np.uint8)
    
    for channel_idx in range(3):
        channel = rgb_data[:, :, channel_idx]
        # Filter out NaN, zero, and negative values for better contrast
        valid_pixels = channel[(~np.isnan(channel)) & (channel > 0)]
        
        if len(valid_pixels) > 0:
            min_val = np.percentile(valid_pixels, percentile_low)
            max_val = np.percentile(valid_pixels, percentile_high)
            
            # Ensure we have a valid range
            if max_val <= min_val:
                max_val = min_val + 1
            
            # Stretch to 0-255 with clipping
            normalized_channel = np.clip(
                (channel - min_val) / (max_val - min_val) * 255, 0, 255
            )
            normalized[:, :, channel_idx] = normalized_channel.astype(np.uint8)
    
    return normalized


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main function to execute color composite visualization workflow."""
    
    print("=" * 80)
    print("COLOR COMPOSITE VISUALIZATION FOR ROI")
    print("=" * 80)
    print()
    
    # Check if input file exists
    if not image_file.exists():
        print(f"Error: Image file not found: {image_file}")
        print("Please ensure Image.tif exists in the data directory.")
        return
    
    # Validate color composites
    if not color_composites or len(color_composites) == 0:
        print("Error: No color composites specified in configuration.")
        return
    
    # Step 1: Load multispectral image
    print("Step 1: Loading multispectral image...")
    bands, profile = load_multispectral_image(image_file)
    print()
    
    # Step 2: Extract ROI
    print("Step 2: Extracting ROI region...")
    center_col, center_row = roi_config['center']
    x_side_length = roi_config['x_side_length']
    y_side_length = roi_config['y_side_length']
    
    height, width = bands.shape[1], bands.shape[2]
    
    # Extract ROI from all bands
    roi_bands = []
    for band_idx in range(bands.shape[0]):
        band_data = bands[band_idx, :, :]
        roi_region, bounds = extract_rectangular_roi(
            band_data, center_row, center_col, x_side_length, y_side_length
        )
        roi_bands.append(roi_region)
    
    # Stack ROI bands: shape (bands, roi_height, roi_width)
    roi_bands_array = np.array(roi_bands)
    
    print(f"  [OK] ROI center: ({center_col}, {center_row})")
    print(f"  [OK] ROI size: {x_side_length} × {y_side_length} pixels")
    print(f"  [OK] Extracted ROI: {roi_bands_array.shape[2]} × {roi_bands_array.shape[1]} pixels")
    print()
    
    # Step 3: Parse composite codes and create RGB composites
    print("Step 3: Creating color composites...")
    
    rgb_composites = []
    composite_descriptions = []
    
    for code in color_composites:
        try:
            band_indices, description = parse_composite_code(code)
            rgb = create_rgb_composite(roi_bands_array, band_indices=band_indices)
            rgb_composites.append(rgb)
            composite_descriptions.append(description)
            print(f"  [OK] {description}")
        except Exception as e:
            print(f"  [ERROR] Failed to create composite {code}: {e}")
            return
    
    print()
    
    # Step 4: Normalize RGB composites
    print("Step 4: Normalizing RGB composites for display...")
    
    normalized_composites = []
    for rgb in rgb_composites:
        normalized = normalize_rgb_for_display(rgb)
        normalized_composites.append(normalized)
    
    print(f"  [OK] Normalized {len(normalized_composites)} composites")
    print()
    
    # Step 5: Create grid visualization
    print("Step 5: Creating grid visualization...")
    
    num_composites = len(normalized_composites)
    
    # Calculate grid dimensions (prefer wider grid)
    if num_composites == 1:
        nrows, ncols = 1, 1
    elif num_composites == 2:
        nrows, ncols = 1, 2
    elif num_composites <= 4:
        nrows, ncols = 2, 2
    elif num_composites <= 6:
        nrows, ncols = 2, 3
    elif num_composites <= 9:
        nrows, ncols = 3, 3
    else:
        # For more than 9, use a wider grid
        ncols = 4
        nrows = (num_composites + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    
    # Handle single subplot case
    if num_composites == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each composite
    for idx, (normalized, description) in enumerate(zip(normalized_composites, composite_descriptions)):
        ax = axes[idx]
        ax.imshow(normalized)
        ax.set_title(description, fontsize=11, fontweight='bold', pad=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_composites, len(axes)):
        axes[idx].axis('off')
    
    # Add overall title
    fig.suptitle(
        f"Color Composites of {roi_config['name']}\n"
        f"Center: ({center_col}, {center_row}), Size: {x_side_length}×{y_side_length} px",
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure
    output_file = output_dir / 'color_composites_roi.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved visualization: {output_file}")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  - Image dimensions: {width} × {height} pixels")
    print(f"  - Number of bands: {bands.shape[0]}")
    print(f"  - ROI: {roi_config['name']}")
    print(f"    Center: ({center_col}, {center_row})")
    print(f"    Size: {x_side_length} × {y_side_length} pixels")
    print(f"    Extracted: {roi_bands_array.shape[2]} × {roi_bands_array.shape[1]} pixels")
    print(f"  - Color composites created: {len(rgb_composites)}")
    for code, desc in zip(color_composites, composite_descriptions):
        print(f"    - {code}: {desc}")
    print(f"  - Output file: {output_file.name}")
    print()


if __name__ == '__main__':
    main()

