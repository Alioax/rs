"""
AOI Comparison Grid Visualization for Change Detection

This script:
1. Loads old (2015) and new (2021) multispectral reflectance images
2. Loads the difference image
3. Creates RGB composites from bands B4, B3, B2
4. Defines 3 AOIs and extracts regions from all images
5. Creates a 3x3 grid visualization: 3 AOIs (rows) × 3 columns (2015 RGB, 2021 RGB, Difference)

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
import json
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
input_dir = data_dir / 'change_detection_images'
output_dir = homework_dir / 'output'
output_dir.mkdir(exist_ok=True, parents=True)
selected_pair_file = data_dir / 'selected_pair.json'

# AOI Configuration
# Define Areas of Interest for comparison visualization
# These are pixel coordinates in the reference image (2021)
# Format: {'center': (col, row), 'side_length': pixels, 'name': 'AOI name'}
# Image dimensions: 7638 x 7877 pixels
aoi_config = [
    {'center': (5400, 4100), 'side_length': 400, 'name': 'AOI 1'},
    {'center': (4000, 4200), 'side_length': 400, 'name': 'AOI 2'},
    {'center': (3650, 6700), 'side_length': 400, 'name': 'AOI 3'},
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
    
    # Extract RGB bands: B4 (index 2), B3 (index 1), B2 (index 0)
    r_band = bands_data[band_indices[0], :, :]  # B4 -> Red
    g_band = bands_data[band_indices[1], :, :]  # B3 -> Green
    b_band = bands_data[band_indices[2], :, :]  # B2 -> Blue
    
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
        valid_pixels = channel[channel > 0]  # Only positive values
        
        if len(valid_pixels) > 0:
            min_val = np.percentile(valid_pixels, percentile_low)
            max_val = np.percentile(valid_pixels, percentile_high)
            
            if max_val > min_val:
                normalized[:, :, channel_idx] = np.clip(
                    (channel - min_val) / (max_val - min_val) * 255, 0, 255
                ).astype(np.uint8)
    
    return normalized


def normalize_difference_for_display(diff_data, percentile_low=2, percentile_high=98):
    """
    Normalize difference data for display using diverging colormap.
    
    Parameters:
    -----------
    diff_data : numpy.ndarray
        2D array of difference values
    percentile_low : float
        Lower percentile for stretching
    percentile_high : float
        Upper percentile for stretching
    
    Returns:
    --------
    numpy.ndarray
        Normalized difference data for colormap display
    tuple
        (vmin, vmax) for colormap scaling
    """
    valid_data = diff_data[~np.isnan(diff_data)]
    
    if len(valid_data) > 0:
        vmin = np.percentile(valid_data, percentile_low)
        vmax = np.percentile(valid_data, percentile_high)
        
        # Ensure symmetric range around zero for diverging colormap
        abs_max = max(abs(vmin), abs(vmax))
        vmin = -abs_max
        vmax = abs_max
    else:
        vmin, vmax = -1, 1
    
    return diff_data, vmin, vmax


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main function to execute AOI comparison visualization workflow."""
    
    print("=" * 80)
    print("AOI COMPARISON GRID VISUALIZATION")
    print("=" * 80)
    print()
    
    # Load selected pair to identify files
    if selected_pair_file.exists():
        with open(selected_pair_file, 'r') as f:
            pair_data = json.load(f)
        
        image_1 = pair_data['selected_pair']['image_1']
        image_2 = pair_data['selected_pair']['image_2']
        
        year_1 = image_1['year']
        year_2 = image_2['year']
        
        if year_1 > year_2:
            newer_scene_id = image_1['scene_id']
            older_scene_id = image_2['scene_id']
            newer_year = year_1
            older_year = year_2
        else:
            newer_scene_id = image_2['scene_id']
            older_scene_id = image_1['scene_id']
            newer_year = year_2
            older_year = year_1
    else:
        # Fallback
        newer_scene_id = "LC81660342021231LGN00"
        older_scene_id = "LC81660342015231LGN01"
        newer_year = 2021
        older_year = 2015
    
    # Define file paths
    older_multispectral = input_dir / f"{older_scene_id}_multispectral_B2-7_reflectance.tif"
    newer_multispectral = input_dir / f"{newer_scene_id}_multispectral_B2-7_reflectance.tif"
    difference_file = input_dir / f"reflectance_difference_{newer_year}-{older_year}.tif"
    
    print(f"Loading images...")
    print(f"  Older ({older_year}): {older_multispectral.name}")
    print(f"  Newer ({newer_year}): {newer_multispectral.name}")
    print(f"  Difference: {difference_file.name}")
    print()
    
    # Step 1: Load multispectral images
    print("Step 1: Loading multispectral reflectance images...")
    
    with rasterio.open(newer_multispectral) as src:
        newer_bands = src.read()  # Shape: [6, height, width]
        newer_profile = src.profile.copy()
        newer_transform = src.transform
        newer_crs = src.crs
        newer_height, newer_width = src.height, src.width
        
        print(f"  [OK] Newer image: {newer_width} × {newer_height} pixels, {src.count} bands")
    
    with rasterio.open(older_multispectral) as src:
        older_bands = src.read()  # Shape: [6, height, width]
        older_profile = src.profile.copy()
        older_transform = src.transform
        older_crs = src.crs
        older_height, older_width = src.height, src.width
        
        print(f"  [OK] Older image: {older_width} × {older_height} pixels, {src.count} bands")
    
    # Step 2: Load difference image
    print("Step 2: Loading difference image...")
    
    with rasterio.open(difference_file) as src:
        difference_data = src.read(1)  # Single band
        diff_transform = src.transform
        diff_crs = src.crs
        diff_height, diff_width = src.height, src.width
        
        print(f"  [OK] Difference image: {diff_width} × {diff_height} pixels")
    
    print()
    
    # Step 3: Create RGB composites
    print("Step 3: Creating RGB composites...")
    
    # Create RGB from bands: B4 (index 2), B3 (index 1), B2 (index 0)
    newer_rgb = create_rgb_composite(newer_bands, band_indices=[2, 1, 0])
    older_rgb = create_rgb_composite(older_bands, band_indices=[2, 1, 0])
    
    print(f"  [OK] Created RGB composites")
    print()
    
    # Step 3.5: Create AOI overview visualization
    print("Step 3.5: Creating AOI overview visualization...")
    
    # Normalize newer RGB for full image display
    newer_rgb_normalized = normalize_rgb_for_display(newer_rgb)
    
    # Calculate image extent
    left = newer_transform[2]
    top = newer_transform[5]
    right = left + newer_width * newer_transform[0]
    bottom = top + newer_height * newer_transform[4]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Display full 2021 RGB image
    ax.imshow(newer_rgb_normalized, extent=[left, right, bottom, top])
    
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
        x_start, y_start = pixel_to_map(col_start, row_start, newer_transform)
        x_end, y_end = pixel_to_map(col_end, row_end, newer_transform)
        
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
    ax.set_title(f'{newer_year} RGB Image with AOI Outlines', fontsize=14, fontweight='bold', pad=10, y=1.02)
    
    # Add legend
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, frameon=False, fontsize=10)
    
    plt.tight_layout()
    overview_output_path = output_dir / 'aoi_overview.png'
    plt.savefig(overview_output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {overview_output_path}")
    plt.close()
    print()
    
    # Step 4: Extract AOI regions
    print("Step 4: Extracting AOI regions...")
    
    # Use newer image as reference for AOI coordinates
    reference_height = newer_height
    reference_width = newer_width
    
    aoi_rgb_regions = {
        'older': [],
        'newer': [],
        'difference': []
    }
    
    for aoi in aoi_config:
        center_col, center_row = aoi['center']
        side_length = aoi['side_length']
        
        # Extract from newer RGB (3D array: height, width, 3)
        # Extract each channel separately and stack
        newer_r_region, bounds = extract_square_region(
            newer_rgb[:, :, 0], int(center_row), int(center_col), side_length
        )
        newer_g_region, _ = extract_square_region(
            newer_rgb[:, :, 1], int(center_row), int(center_col), side_length
        )
        newer_b_region, _ = extract_square_region(
            newer_rgb[:, :, 2], int(center_row), int(center_col), side_length
        )
        newer_rgb_aoi = np.dstack([newer_r_region, newer_g_region, newer_b_region])
        aoi_rgb_regions['newer'].append(newer_rgb_aoi)
        
        # Extract from older RGB (use same coordinates)
        older_r_region, _ = extract_square_region(
            older_rgb[:, :, 0], int(center_row), int(center_col), side_length
        )
        older_g_region, _ = extract_square_region(
            older_rgb[:, :, 1], int(center_row), int(center_col), side_length
        )
        older_b_region, _ = extract_square_region(
            older_rgb[:, :, 2], int(center_row), int(center_col), side_length
        )
        # Ensure same shape as newer
        target_shape = newer_rgb_aoi.shape[:2]
        if older_r_region.shape != target_shape:
            # Simple cropping/padding to match
            h, w = older_r_region.shape
            th, tw = target_shape
            if h < th or w < tw:
                # Pad
                pad_h = max(0, th - h)
                pad_w = max(0, tw - w)
                older_r_region = np.pad(older_r_region, ((0, pad_h), (0, pad_w)), mode='edge')
                older_g_region = np.pad(older_g_region, ((0, pad_h), (0, pad_w)), mode='edge')
                older_b_region = np.pad(older_b_region, ((0, pad_h), (0, pad_w)), mode='edge')
            # Crop to match
            older_r_region = older_r_region[:th, :tw]
            older_g_region = older_g_region[:th, :tw]
            older_b_region = older_b_region[:th, :tw]
        older_rgb_aoi = np.dstack([older_r_region, older_g_region, older_b_region])
        aoi_rgb_regions['older'].append(older_rgb_aoi)
        
        # Extract from difference image
        diff_region, _ = extract_square_region(
            difference_data, int(center_row), int(center_col), side_length
        )
        # Ensure same shape
        if diff_region.shape != target_shape:
            h, w = diff_region.shape
            th, tw = target_shape
            if h < th or w < tw:
                pad_h = max(0, th - h)
                pad_w = max(0, tw - w)
                diff_region = np.pad(diff_region, ((0, pad_h), (0, pad_w)), mode='edge')
            diff_region = diff_region[:th, :tw]
        aoi_rgb_regions['difference'].append(diff_region)
        
        print(f"  [OK] Extracted {aoi['name']}")
    
    print()
    
    # Step 5: Normalize for display
    print("Step 5: Normalizing images for display...")
    
    # Normalize RGB composites (consistent scaling per column)
    # Collect all regions for each column to compute global min/max per channel
    older_all_r = []
    older_all_g = []
    older_all_b = []
    newer_all_r = []
    newer_all_g = []
    newer_all_b = []
    
    for reg in aoi_rgb_regions['older']:
        older_all_r.extend(reg[:, :, 0].flatten())
        older_all_g.extend(reg[:, :, 1].flatten())
        older_all_b.extend(reg[:, :, 2].flatten())
    
    for reg in aoi_rgb_regions['newer']:
        newer_all_r.extend(reg[:, :, 0].flatten())
        newer_all_g.extend(reg[:, :, 1].flatten())
        newer_all_b.extend(reg[:, :, 2].flatten())
    
    # Calculate percentile ranges for consistent scaling
    older_percentiles = {
        'R': (np.percentile(older_all_r, 2), np.percentile(older_all_r, 98)),
        'G': (np.percentile(older_all_g, 2), np.percentile(older_all_g, 98)),
        'B': (np.percentile(older_all_b, 2), np.percentile(older_all_b, 98))
    }
    newer_percentiles = {
        'R': (np.percentile(newer_all_r, 2), np.percentile(newer_all_r, 98)),
        'G': (np.percentile(newer_all_g, 2), np.percentile(newer_all_g, 98)),
        'B': (np.percentile(newer_all_b, 2), np.percentile(newer_all_b, 98))
    }
    
    # Normalize difference
    diff_all = np.concatenate([reg.flatten() for reg in aoi_rgb_regions['difference']])
    
    # Normalize difference for diverging colormap
    diff_valid = diff_all[~np.isnan(diff_all)]
    if len(diff_valid) > 0:
        diff_vmin = np.percentile(diff_valid, 2)
        diff_vmax = np.percentile(diff_valid, 98)
        diff_abs_max = max(abs(diff_vmin), abs(diff_vmax))
        diff_vmin, diff_vmax = -diff_abs_max, diff_abs_max
    else:
        diff_vmin, diff_vmax = -0.1, 0.1
    
    print(f"  [OK] Normalization ranges computed")
    print()
    
    # Step 5.5: Create AOI overview visualization for difference map (using same normalization as grid)
    print("Step 5.5: Creating AOI overview visualization (difference map)...")
    
    # Calculate difference map extent (should be same as newer image)
    diff_left = diff_transform[2]
    diff_top = diff_transform[5]
    diff_right = diff_left + diff_width * diff_transform[0]
    diff_bottom = diff_top + diff_height * diff_transform[4]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Display difference map with same colormap and normalization as grid
    im = ax.imshow(difference_data, cmap='RdBu_r', extent=[diff_left, diff_right, diff_bottom, diff_top],
                   vmin=diff_vmin, vmax=diff_vmax)
    
    # Draw AOI outlines
    for aoi_idx, aoi in enumerate(aoi_config):
        center_col, center_row = aoi['center']
        side_length = aoi['side_length']
        
        half_side = side_length // 2
        row_start = center_row - half_side
        row_end = center_row + half_side + (side_length % 2)
        col_start = center_col - half_side
        col_end = center_col + half_side + (side_length % 2)
        
        # Convert to map coordinates (use diff_transform since we're displaying difference map)
        x_start, y_start = pixel_to_map(col_start, row_start, diff_transform)
        x_end, y_end = pixel_to_map(col_end, row_end, diff_transform)
        
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
    ax.set_xlim(diff_left, diff_right)
    ax.set_ylim(diff_bottom, diff_top)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_title(f'Reflectance Difference ({newer_year} - {older_year}) with AOI Outlines', 
                 fontsize=14, fontweight='bold', pad=10, y=1.02)
    
    # Add colorbar (horizontal, below the image) - using same values as grid
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label(f'Reflectance Difference ({newer_year} - {older_year})', fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    # Format tick labels to show reasonable precision
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    # Add legend
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, frameon=False, fontsize=10)
    
    plt.tight_layout()
    diff_overview_output_path = output_dir / 'aoi_overview_difference.png'
    plt.savefig(diff_overview_output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {diff_overview_output_path}")
    plt.close()
    print()
    
    # Step 6: Create 3x3 grid visualization
    print("Step 6: Creating 3x3 comparison grid...")
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    fig.suptitle(f'Change Detection Comparison: {older_year} vs {newer_year}', 
                 fontsize=16, fontweight='bold', y=1.05)
    
    # Store the imshow object from difference column for colorbar
    diff_imshow = None
    
    # Process each AOI (rows)
    for row_idx, aoi in enumerate(aoi_config):
        # Column 1: Older RGB (2015)
        ax = axes[row_idx, 0]
        older_rgb_aoi = aoi_rgb_regions['older'][row_idx]
        older_normalized = np.zeros_like(older_rgb_aoi, dtype=np.uint8)
        for ch_idx, color in enumerate(['R', 'G', 'B']):
            channel = older_rgb_aoi[:, :, ch_idx]
            min_val, max_val = older_percentiles[color]
            if max_val > min_val:
                normalized_ch = np.clip((channel - min_val) / (max_val - min_val) * 255, 0, 255)
                older_normalized[:, :, ch_idx] = normalized_ch.astype(np.uint8)
        ax.imshow(older_normalized, aspect='equal', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(aoi_colors[row_idx])
            spine.set_linewidth(3)
        if row_idx == 0:
            ax.set_title(f"{older_year} RGB", fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel(aoi['name'], fontsize=12, fontweight='bold', rotation=90, labelpad=20)
        
        # Column 2: Newer RGB (2021)
        ax = axes[row_idx, 1]
        newer_rgb_aoi = aoi_rgb_regions['newer'][row_idx]
        newer_normalized = np.zeros_like(newer_rgb_aoi, dtype=np.uint8)
        for ch_idx, color in enumerate(['R', 'G', 'B']):
            channel = newer_rgb_aoi[:, :, ch_idx]
            min_val, max_val = newer_percentiles[color]
            if max_val > min_val:
                normalized_ch = np.clip((channel - min_val) / (max_val - min_val) * 255, 0, 255)
                newer_normalized[:, :, ch_idx] = normalized_ch.astype(np.uint8)
        ax.imshow(newer_normalized, aspect='equal', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(aoi_colors[row_idx])
            spine.set_linewidth(3)
        if row_idx == 0:
            ax.set_title(f"{newer_year} RGB", fontsize=12, fontweight='bold', pad=10)
        
        # Column 3: Difference
        ax = axes[row_idx, 2]
        diff_aoi = aoi_rgb_regions['difference'][row_idx]
        im = ax.imshow(diff_aoi, cmap='RdBu_r', aspect='equal', interpolation='nearest',
                      vmin=diff_vmin, vmax=diff_vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(aoi_colors[row_idx])
            spine.set_linewidth(3)
        if row_idx == 0:
            ax.set_title("Difference", fontsize=12, fontweight='bold', pad=10)
            # Store imshow object for colorbar (use first row's difference)
            diff_imshow = im
    
    # Apply tight_layout first to get proper positions
    plt.tight_layout()
    
    # Add horizontal colorbar above column titles (after tight_layout to get correct positions)
    if diff_imshow is not None:
        # Get positions of leftmost and rightmost columns in the top row
        left_ax = axes[0, 0]
        right_ax = axes[0, 2]
        pos_left = left_ax.get_position()
        pos_right = right_ax.get_position()
        
        # Calculate full width spanning all 3 columns
        cbar_x0 = pos_left.x0  # Left edge of first column
        cbar_width = pos_right.x1 - pos_left.x0  # Full width from left to right edge
        
        # Position colorbar between main title and column titles
        # Get the top position of the first row plots
        top_row_y = pos_left.y1  # Top of the top row plots
        
        # Position colorbar above the plots but below main title
        # Leave some space for column titles (they have pad=10)
        cbar_y = top_row_y + 0.06  # Above the plots, accounting for title space
        cbar_height = 0.015
        
        # Create colorbar axis spanning all 3 columns
        cbar_ax = fig.add_axes([
            cbar_x0,  # Left edge of first column
            cbar_y,  # Above the plots
            cbar_width,  # Full width of all 3 columns
            cbar_height  # Height of colorbar
        ])
        
        # Create horizontal colorbar
        cbar = fig.colorbar(diff_imshow, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(f'Reflectance Difference ({newer_year} - {older_year})', fontsize=12, labelpad=5)
        cbar.ax.tick_params(labelsize=10)
        # Format tick labels to show reasonable precision
        cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    output_path = output_dir / 'aoi_comparison_3x3.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()
    print()
    
    # Final summary
    print("=" * 80)
    print("AOI COMPARISON VISUALIZATION SUMMARY")
    print("=" * 80)
    print(f"\nImages compared:")
    print(f"  - {older_year} RGB: {older_width} × {older_height} pixels")
    print(f"  - {newer_year} RGB: {newer_width} × {newer_height} pixels")
    print(f"  - Difference: {diff_width} × {diff_height} pixels")
    
    print(f"\nAOIs analyzed: {len(aoi_config)}")
    for aoi in aoi_config:
        print(f"  - {aoi['name']}: center=({aoi['center'][0]}, {aoi['center'][1]}), size={aoi['side_length']}px")
    
    print(f"\nOutput files:")
    print(f"  - AOI overview (RGB): aoi_overview.png")
    print(f"  - AOI overview (Difference): aoi_overview_difference.png")
    print(f"  - Comparison grid: aoi_comparison_3x3.png")
    
    print("\n" + "=" * 80)
    print("[OK] Visualization completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

