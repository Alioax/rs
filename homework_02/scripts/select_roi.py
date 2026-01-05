"""
ROI Selection Visualization

This script:
1. Loads multispectral Landsat 8 image (bands 2-7) from Image.tif
2. Creates a default RGB composite (432: B4, B3, B2) for visualization
3. Plots the full image with a configurable rectangular ROI outline

Author: Ali Haghighi
Course: Remote Sensing
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rasterio
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib parameters for high-quality plots
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (14, 12)

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

# ROI Configuration
# Define rectangular Region of Interest
# Format: {'center': (col, row), 'x_side_length': width, 'y_side_length': height, 'name': 'ROI name'}
roi_config = {
    'center': (2500, 4000),  # (col, row) in pixels
    'x_side_length': 1600,    # width in pixels
    'y_side_length': 900,    # height in pixels
    'name': 'ROI 1'
}

# Default RGB composite for visualization (432 = B4, B3, B2)
default_rgb_bands = [2, 1, 0]  # B4 (index 2), B3 (index 1), B2 (index 0)

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
    rasterio.transform.Affine
        Geotransform
    """
    with rasterio.open(image_path) as src:
        bands = src.read()  # Shape: [bands, height, width]
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        
        print(f"  [OK] Loaded image: {src.width} × {src.height} pixels, {src.count} bands")
        print(f"  [OK] CRS: {crs}")
        print(f"  [OK] Data type: {bands.dtype}")
        
        return bands, profile, transform, crs


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


def normalize_rgb_for_display(rgb_data, percentile_low=2, percentile_high=98):
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
        valid_pixels = channel[channel > 0]  # Only positive values (matches homework 7)
        
        if len(valid_pixels) > 0:
            min_val = np.percentile(valid_pixels, percentile_low)
            max_val = np.percentile(valid_pixels, percentile_high)
            
            if max_val > min_val:
                normalized[:, :, channel_idx] = np.clip(
                    (channel - min_val) / (max_val - min_val) * 255, 0, 255
                ).astype(np.uint8)
    
    return normalized


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main function to execute ROI selection visualization workflow."""
    
    print("=" * 80)
    print("ROI SELECTION VISUALIZATION")
    print("=" * 80)
    print()
    
    # Check if input file exists
    if not image_file.exists():
        print(f"Error: Image file not found: {image_file}")
        print("Please ensure Image.tif exists in the data directory.")
        return
    
    # Step 1: Load multispectral image
    print("Step 1: Loading multispectral image...")
    bands, profile, transform, crs = load_multispectral_image(image_file)
    print()
    
    # Step 2: Create RGB composite
    print("Step 2: Creating RGB composite...")
    rgb = create_rgb_composite(bands, band_indices=default_rgb_bands)
    print(f"  [OK] Created RGB composite (432: B4, B3, B2)")
    print()
    
    # Step 3: Normalize RGB for display
    print("Step 3: Normalizing RGB for display...")
    rgb_normalized = normalize_rgb_for_display(rgb)
    print(f"  [OK] Normalized RGB composite")
    print()
    
    # Step 4: Extract ROI bounds
    print("Step 4: Extracting ROI bounds...")
    center_col, center_row = roi_config['center']
    x_side_length = roi_config['x_side_length']
    y_side_length = roi_config['y_side_length']
    
    height, width = rgb.shape[:2]
    
    # Calculate ROI bounds
    half_x = x_side_length // 2
    half_y = y_side_length // 2
    
    col_start = center_col - half_x
    col_end = center_col + half_x + (x_side_length % 2)
    row_start = center_row - half_y
    row_end = center_row + half_y + (y_side_length % 2)
    
    # Clamp to image bounds
    col_start = max(0, int(col_start))
    col_end = min(width, int(col_end))
    row_start = max(0, int(row_start))
    row_end = min(height, int(row_end))
    
    # Calculate rectangle position and size for matplotlib
    rect_x = col_start
    rect_y = row_start
    rect_width = col_end - col_start
    rect_height = row_end - row_start
    
    print(f"  [OK] ROI center: ({center_col}, {center_row})")
    print(f"  [OK] ROI size: {x_side_length} × {y_side_length} pixels")
    print(f"  [OK] ROI bounds: col [{col_start}, {col_end}], row [{row_start}, {row_end}]")
    print()
    
    # Step 5: Create visualization
    print("Step 5: Creating visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Display RGB image
    ax.imshow(rgb_normalized, extent=[0, width, height, 0])
    
    # Draw ROI rectangle
    rect = patches.Rectangle(
        (rect_x, rect_y),
        rect_width,
        rect_height,
        linewidth=3,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    
    # Add annotation at the bottom of the ROI
    annotation_x = center_col
    annotation_y = row_end + 256  # Position below the ROI rectangle
    ax.annotate(
        'Region of Interest',
        xy=(annotation_x, annotation_y),
        ha='center',
        va='bottom',
        fontsize=18,
        fontweight='bold',
        color='red',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='None', edgecolor='None', linewidth=2)
    )
    
    # Remove axes, labels, grid, and title
    ax.axis('off')
    
    # Save figure
    output_file = output_dir / 'roi_selection.png'
    plt.tight_layout()
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
    print(f"  - Output file: {output_file.name}")
    print()


if __name__ == '__main__':
    main()

