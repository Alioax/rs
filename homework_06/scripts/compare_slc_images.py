"""
SLC Error Image Comparison

This script compares three SLC (Scan Line Corrector) error images:
- SLC error (original)
- SLC self-fill (filled with itself)
- SLC two band gap fill (filled with another reference picture)

It generates two visualizations:
1. AOI overview showing the 3 AOIs on the SLC error image
2. 3x3 comparison grid with 3 images (rows) and 3 AOIs (columns)

Author: Ali Haghighi
Course: Remote Sensing, Fall 2025
Instructor: Dr. Parviz Fatehi
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
plt.rcParams['figure.figsize'] = (12, 10)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define paths
script_dir = Path(__file__).parent
homework_06_dir = script_dir.parent

# Input paths
slc_dir = homework_06_dir / 'data' / 'SLC'

# Output directory
output_dir = homework_06_dir / 'output'
output_dir.mkdir(exist_ok=True, parents=True)

# AOI Configuration
# Define Areas of Interest for comparison visualization
# These are pixel coordinates in the SLC error image
# Format: {'center': (col, row), 'side_length': pixels, 'name': 'AOI name'}
# Default coordinates - can be adjusted based on SLC image dimensions
aoi_config = [
    {'center': (1600, 1200), 'side_length': 400, 'name': 'AOI 1'},
    {'center': (1600, 800), 'side_length': 200, 'name': 'AOI 2'},
    {'center': (6000, 3800), 'side_length': 500, 'name': 'AOI 3'},
]

# AOI colors for visualization
aoi_colors = ['red', 'lime', 'cyan']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def pixel_to_map(col, row, transform):
    """Convert pixel coordinates to map coordinates."""
    x = transform[0] * col + transform[2]
    y = transform[4] * row + transform[5]
    return x, y


def extract_square_region(band_data, center_row, center_col, side_length):
    """
    Extract a square region from a band array.
    
    Parameters:
    -----------
    band_data : numpy.ndarray
        2D array of band data
    center_row : int
        Row coordinate of center pixel
    center_col : int
        Column coordinate of center pixel
    side_length : int
        Side length of square region in pixels
    
    Returns:
    --------
    numpy.ndarray
        Extracted square region
    tuple
        (row_start, row_end, col_start, col_end) - actual bounds used
    """
    height, width = band_data.shape
    
    # Calculate half side length
    half_side = side_length // 2
    
    # Calculate bounding box
    row_start = center_row - half_side
    row_end = center_row + half_side + (side_length % 2)
    col_start = center_col - half_side
    col_end = center_col + half_side + (side_length % 2)
    
    # Clamp to image bounds
    row_start = max(0, int(row_start))
    row_end = min(height, int(row_end))
    col_start = max(0, int(col_start))
    col_end = min(width, int(col_end))
    
    # Extract region
    region = band_data[row_start:row_end, col_start:col_end]
    
    return region, (row_start, row_end, col_start, col_end)


def normalize_image_for_display(image_data, percentile_low=2, percentile_high=98):
    """
    Normalize image data for display using percentile-based stretching.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        Image data (can be 2D grayscale or 3D RGB)
    percentile_low : float
        Lower percentile for stretching (default 2)
    percentile_high : float
        Upper percentile for stretching (default 98)
    
    Returns:
    --------
    numpy.ndarray
        Normalized image data (uint8, 0-255)
    """
    if len(image_data.shape) == 2:
        # Grayscale image
        valid_pixels = image_data[image_data > 0]
        if len(valid_pixels) > 0:
            min_val = np.percentile(valid_pixels, percentile_low)
            max_val = np.percentile(valid_pixels, percentile_high)
            if max_val > min_val:
                normalized = np.clip((image_data - min_val) / (max_val - min_val) * 255, 0, 255)
            else:
                normalized = np.zeros_like(image_data, dtype=np.uint8)
        else:
            normalized = np.zeros_like(image_data, dtype=np.uint8)
        return normalized.astype(np.uint8)
    
    elif len(image_data.shape) == 3:
        # Multi-band image (RGB)
        normalized = np.zeros_like(image_data, dtype=np.uint8)
        for i in range(image_data.shape[2]):
            band = image_data[:, :, i]
            valid_pixels = band[band > 0]
            if len(valid_pixels) > 0:
                min_val = np.percentile(valid_pixels, percentile_low)
                max_val = np.percentile(valid_pixels, percentile_high)
                if max_val > min_val:
                    normalized[:, :, i] = np.clip((band - min_val) / (max_val - min_val) * 255, 0, 255)
        return normalized
    
    else:
        raise ValueError(f"Unsupported image shape: {image_data.shape}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main function to execute SLC image comparison workflow."""
    
    print("=" * 80)
    print("SLC ERROR IMAGE COMPARISON")
    print("=" * 80)
    print()
    
    # Step 1: Load SLC images
    print("Step 1: Loading SLC images...")
    
    slc_images = {}
    image_files = {
        'error': slc_dir / 'SLC error.tif',
        'self_fill': slc_dir / 'SLC self-fill.tif',
        'two_band_gap_fill': slc_dir / 'SLC two band gap fill.tif'
    }
    
    for image_key, file_path in image_files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"SLC image not found: {file_path}")
        
        with rasterio.open(file_path) as src:
            # Read all bands (assuming multi-band RGB)
            if src.count >= 3:
                # Read as RGB
                slc_images[image_key] = {
                    'data': np.dstack([src.read(i+1) for i in range(3)]),  # Stack bands as RGB
                    'profile': src.profile.copy(),
                    'transform': src.transform,
                    'crs': src.crs,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count
                }
            elif src.count == 1:
                # Single band - convert to grayscale RGB
                band_data = src.read(1)
                slc_images[image_key] = {
                    'data': np.dstack([band_data, band_data, band_data]),  # Duplicate for RGB
                    'profile': src.profile.copy(),
                    'transform': src.transform,
                    'crs': src.crs,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count
                }
            else:
                # Use first 3 bands if more than 3
                slc_images[image_key] = {
                    'data': np.dstack([src.read(i+1) for i in range(min(3, src.count))]),
                    'profile': src.profile.copy(),
                    'transform': src.transform,
                    'crs': src.crs,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count
                }
        
        print(f"  [OK] Loaded {image_key}: {slc_images[image_key]['width']} x {slc_images[image_key]['height']} pixels, {slc_images[image_key]['count']} bands")
    
    print()
    
    # Use SLC error as reference for AOI coordinates
    reference_image = slc_images['error']
    ref_transform = reference_image['transform']
    ref_width = reference_image['width']
    ref_height = reference_image['height']
    
    # Step 2: Create AOI overview visualization
    print("Step 2: Creating AOI overview visualization...")
    
    # Calculate image extent
    left = ref_transform[2]
    top = ref_transform[5]
    right = left + ref_width * ref_transform[0]
    bottom = top + ref_height * ref_transform[4]
    
    # Normalize reference image for display
    ref_rgb_normalized = normalize_image_for_display(reference_image['data'])
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Display SLC error image
    ax.imshow(ref_rgb_normalized, extent=[left, right, bottom, top])
    
    # Draw AOI outlines
    for aoi_idx, aoi in enumerate(aoi_config):
        center_col, center_row = aoi['center']
        side_length = aoi['side_length']
        
        half_side = side_length // 2
        row_start = center_row - half_side
        row_end = center_row + half_side + (side_length % 2)
        col_start = center_col - half_side
        col_end = center_col + half_side + (side_length % 2)
        
        # Convert to map coordinates
        x_start, y_start = pixel_to_map(col_start, row_start, ref_transform)
        x_end, y_end = pixel_to_map(col_end, row_end, ref_transform)
        
        rect_width = x_end - x_start
        rect_height = y_end - y_start
        
        rect = patches.Rectangle(
            (x_start, y_start),
            rect_width,
            rect_height,
            linewidth=2.5,
            edgecolor=aoi_colors[aoi_idx],
            facecolor='none',
            label=aoi['name']
        )
        ax.add_patch(rect)
    
    # Set axis limits
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_title('SLC Error Image with AOI Outlines', fontsize=14, fontweight='bold', pad=10, y=1.02)
    
    # Add legend
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, frameon=False, fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'slc_aoi_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()
    print()
    
    # Step 3: Create 3x3 comparison grid
    print("Step 3: Creating 3x3 comparison grid...")
    
    # Extract AOI regions from each image
    image_names = ['SLC Error', 'SLC Self-Fill', 'SLC Two Band Gap Fill']
    image_keys = ['error', 'self_fill', 'two_band_gap_fill']
    
    # Store RGB regions as separate R, G, B channels (like in image_to_image_registration.py)
    aoi_rgb_regions_by_image = {}
    
    for image_key in image_keys:
        if image_key not in slc_images:
            continue
        
        image_data = slc_images[image_key]
        aoi_rgb_regions_by_image[image_key] = []
        
        for aoi in aoi_config:
            center_col, center_row = aoi['center']
            side_length = aoi['side_length']
            
            # Extract RGB region as separate R, G, B channels
            rgb_regions = {}
            ref_band = image_data['data'][:, :, 0]  # Use first band for shape reference
            
            if 0 <= center_row < ref_band.shape[0] and 0 <= center_col < ref_band.shape[1]:
                # Extract each RGB channel separately
                for channel_idx, color in enumerate(['R', 'G', 'B']):
                    if channel_idx < image_data['data'].shape[2]:
                        band_data = image_data['data'][:, :, channel_idx]
                        region, _ = extract_square_region(band_data, int(center_row), int(center_col), side_length)
                        rgb_regions[color] = region
                    else:
                        # If fewer than 3 bands, duplicate the last band
                        band_data = image_data['data'][:, :, -1]
                        region, _ = extract_square_region(band_data, int(center_row), int(center_col), side_length)
                        rgb_regions[color] = region
            else:
                # Out of bounds - create empty region
                empty_region = np.zeros((side_length, side_length))
                rgb_regions = {
                    'R': empty_region,
                    'G': empty_region,
                    'B': empty_region
                }
            
            aoi_rgb_regions_by_image[image_key].append(rgb_regions)
    
    # Calculate percentile-based min/max for each band and each AOI (column) across all images (rows)
    # This ensures each column and each color channel has consistent scaling, avoiding outliers
    aoi_band_min_max = []  # List of dicts: [{'R': (min, max), 'G': (min, max), 'B': (min, max)}, ...]
    num_aois = len(aoi_config)
    percentile_low = 2   # Lower percentile to avoid dark outliers
    percentile_high = 98  # Upper percentile to avoid bright outliers
    
    for aoi_idx in range(num_aois):
        # Collect all valid pixel values for each band across all images
        band_all_values = {'R': [], 'G': [], 'B': []}
        
        for image_key in image_keys:
            if image_key in aoi_rgb_regions_by_image:
                if aoi_idx < len(aoi_rgb_regions_by_image[image_key]):
                    rgb_regions = aoi_rgb_regions_by_image[image_key][aoi_idx]
                    for color in ['R', 'G', 'B']:
                        region = rgb_regions[color]
                        if region.size > 0:
                            valid_data = region[region > 0]  # Only positive values
                            if len(valid_data) > 0:
                                band_all_values[color].extend(valid_data.flatten())
        
        # Calculate percentile-based min/max for each band
        aoi_band_min_max_dict = {}
        for color in ['R', 'G', 'B']:
            if len(band_all_values[color]) > 0:
                # Use percentiles to avoid outliers
                band_min = np.percentile(band_all_values[color], percentile_low)
                band_max = np.percentile(band_all_values[color], percentile_high)
                # Ensure we have a valid range
                if band_max <= band_min:
                    band_max = band_min + 1
            else:
                band_min, band_max = 0, 255
            aoi_band_min_max_dict[color] = (band_min, band_max)
        
        aoi_band_min_max.append(aoi_band_min_max_dict)
    
    # Create 3x3 grid figure
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    
    # Process each image (rows)
    for row_idx, (image_name, image_key) in enumerate(zip(image_names, image_keys)):
        if image_key not in aoi_rgb_regions_by_image:
            # Skip if image not available
            for col_idx in range(3):
                ax = axes[row_idx, col_idx]
                ax.axis('off')
            continue
        
        rgb_regions_list = aoi_rgb_regions_by_image[image_key]
        
        # Process each AOI (columns)
        for col_idx, rgb_regions in enumerate(rgb_regions_list):
            ax = axes[row_idx, col_idx]
            
            # Get min/max for this AOI and each band
            aoi_band_ranges = aoi_band_min_max[col_idx]
            
            # Normalize each band using the same scale for this AOI
            rgb_normalized = np.zeros((rgb_regions['R'].shape[0], rgb_regions['R'].shape[1], 3), dtype=np.uint8)
            
            for color_idx, color in enumerate(['R', 'G', 'B']):
                band_data = rgb_regions[color].astype(np.float64)  # Use float for calculations
                band_min, band_max = aoi_band_ranges[color]
                
                if band_max > band_min and band_data.size > 0:
                    # Create mask for valid (non-zero) pixels
                    valid_mask = band_data > 0
                    
                    # Initialize normalized array with zeros
                    normalized = np.zeros_like(band_data)
                    
                    # Only normalize valid (non-zero) pixels
                    if np.any(valid_mask):
                        valid_data = band_data[valid_mask]
                        # Normalize valid pixels to 0-255 range
                        normalized_valid = (valid_data - band_min) / (band_max - band_min) * 255
                        normalized_valid = np.clip(normalized_valid, 0, 255)
                        normalized[valid_mask] = normalized_valid
                    
                    rgb_normalized[:, :, color_idx] = normalized.astype(np.uint8)
                else:
                    # Fallback: use the band data directly if normalization fails
                    if band_data.size > 0:
                        # Simple linear stretch if percentiles failed
                        valid_mask = band_data > 0
                        normalized = np.zeros_like(band_data)
                        
                        if np.any(valid_mask):
                            band_data_clean = band_data[valid_mask]
                            simple_min = band_data_clean.min()
                            simple_max = band_data_clean.max()
                            if simple_max > simple_min:
                                normalized_valid = (band_data_clean - simple_min) / (simple_max - simple_min) * 255
                                normalized_valid = np.clip(normalized_valid, 0, 255)
                                normalized[valid_mask] = normalized_valid
                        
                        rgb_normalized[:, :, color_idx] = normalized.astype(np.uint8)
                    else:
                        rgb_normalized[:, :, color_idx] = np.zeros_like(band_data, dtype=np.uint8)
            
            # Display RGB image
            if rgb_normalized.size > 0:
                ax.imshow(rgb_normalized, aspect='equal', interpolation='nearest')
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colored border matching AOI color
            for spine in ax.spines.values():
                spine.set_edgecolor(aoi_colors[col_idx])
                spine.set_linewidth(3)
            
            # Add labels
            if row_idx == 0:
                ax.set_title(f"{aoi_config[col_idx]['name']}", fontsize=12, fontweight='bold', pad=10)
            
            if col_idx == 0:
                ax.set_ylabel(image_name, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'slc_comparison_3x3.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()
    print()
    
    # Final summary
    print("=" * 80)
    print("SLC IMAGE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\nImages compared:")
    for image_name, image_key in zip(image_names, image_keys):
        if image_key in slc_images:
            img = slc_images[image_key]
            print(f"  - {image_name}: {img['width']} x {img['height']} pixels")
    
    print(f"\nAOIs analyzed: {len(aoi_config)}")
    for aoi in aoi_config:
        print(f"  - {aoi['name']}: center=({aoi['center'][0]}, {aoi['center'][1]}), size={aoi['side_length']}px")
    
    print(f"\nOutput files:")
    print(f"  - AOI overview: slc_aoi_overview.png")
    print(f"  - Comparison grid: slc_comparison_3x3.png")
    
    print("\n" + "=" * 80)
    print("[OK] All processing completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

