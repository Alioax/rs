"""
Diffusion Comparison Visualization

This script:
1. Loads before and after diffusion multispectral images (bands 2-7)
2. Creates RGB composites from bands B4, B3, B2
3. Defines AOIs and extracts regions from both images
4. Creates a 2xN grid visualization: 2 rows (Before/After) × N columns (AOIs)

Author: Ali Haghighi
Course: Remote Sensing
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rasterio
from rasterio.warp import reproject, Resampling
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
input_dir = data_dir / 'diffusion'
output_dir = homework_dir / 'output'
output_dir.mkdir(exist_ok=True, parents=True)

# AOI Configuration
# Define Areas of Interest for comparison visualization
# These are pixel coordinates in the reference image (before diffusion)
# Format: {'center': (col, row), 'side_length': pixels, 'name': 'AOI name'}
aoi_config = [
    {'center': (3800, 6750), 'side_length': 100, 'name': 'AOI 1'},
    {'center': (5500, 4250), 'side_length': 100, 'name': 'AOI 2'},
    {'center': (2500, 5000), 'side_length': 150, 'name': 'AOI 3'},
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


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main function to execute diffusion comparison visualization workflow."""
    
    print("=" * 80)
    print("DIFFUSION COMPARISON VISUALIZATION")
    print("=" * 80)
    print()
    
    # Define file paths
    before_file = input_dir / 'Before Diffusion.tif'
    after_file = input_dir / 'Diffusion.tif'
    
    print(f"Loading images...")
    print(f"  Before: {before_file.name}")
    print(f"  After: {after_file.name}")
    print()
    
    # Step 1: Load multispectral images
    print("Step 1: Loading multispectral images...")
    
    with rasterio.open(before_file) as src:
        before_bands = src.read()  # Shape: [6, height, width]
        before_profile = src.profile.copy()
        before_transform = src.transform
        before_crs = src.crs
        before_height, before_width = src.height, src.width
        
        print(f"  [OK] Before image: {before_width} × {before_height} pixels, {src.count} bands")
        print(f"      CRS: {before_crs}")
        print(f"      Pixel size: {abs(before_transform[0]):.1f}m × {abs(before_transform[4]):.1f}m")
    
    with rasterio.open(after_file) as src:
        after_bands_raw = src.read()  # Shape: [6, height, width]
        after_profile = src.profile.copy()
        after_transform_raw = src.transform
        after_crs_raw = src.crs
        after_height_raw, after_width_raw = src.height, src.width
        
        print(f"  [OK] After image (raw): {after_width_raw} × {after_height_raw} pixels, {src.count} bands")
        print(f"      CRS: {after_crs_raw}")
        print(f"      Pixel size: {abs(after_transform_raw[0]):.1f}m × {abs(after_transform_raw[4]):.1f}m")
    
    print()
    
    # Step 1.5: Check and handle spatial alignment
    print("Step 1.5: Checking spatial alignment...")
    print("  Strategy: Use AFTER image (15m) as reference grid to preserve higher resolution")
    print("           Reproject BEFORE image (30m) to match AFTER image grid")
    
    crs_match = before_crs == after_crs_raw
    transform_match = before_transform == after_transform_raw
    shape_match = (before_height == after_height_raw) and (before_width == after_width_raw)
    
    print(f"  CRS match: {crs_match}")
    print(f"  Transform match: {transform_match}")
    print(f"  Shape match: {shape_match}")
    
    # Always use after image as reference (higher resolution)
    # Reproject before image to match after image's grid
    after_bands = after_bands_raw
    after_transform = after_transform_raw
    after_crs = after_crs_raw
    after_height = after_height_raw
    after_width = after_width_raw
    
    if crs_match and transform_match and shape_match:
        print("  [OK] Images are perfectly aligned. Using direct pixel coordinates.")
        before_bands_aligned = before_bands
        before_transform_aligned = before_transform
    else:
        print("  [INFO] Reprojecting/resampling BEFORE image to match AFTER image grid (15m)...")
        print("  This preserves the higher resolution detail in the after image.")
        print("  This may take a moment...")
        
        # Reproject/resample BEFORE image to match AFTER image's grid (higher resolution)
        before_bands_aligned = np.empty((6, after_height, after_width), dtype=before_bands.dtype)
        
        for band_idx in range(6):
            reproject(
                source=before_bands[band_idx, :, :].astype(np.float32),
                destination=before_bands_aligned[band_idx, :, :],
                src_transform=before_transform,
                src_crs=before_crs,
                dst_transform=after_transform,
                dst_crs=after_crs,
                resampling=Resampling.bilinear
            )
        
        before_transform_aligned = after_transform
        
        print(f"  [OK] Before image reprojected: {after_width} × {after_height} pixels")
        print(f"      Now matches after image grid (15m resolution)")
    
    print()
    
    # Step 2: Create RGB composites
    print("Step 2: Creating RGB composites...")
    
    # Create RGB from bands: B4 (index 2), B3 (index 1), B2 (index 0)
    # Both images now use the same grid (after image's 15m grid)
    before_rgb = create_rgb_composite(before_bands_aligned, band_indices=[2, 1, 0])
    after_rgb = create_rgb_composite(after_bands, band_indices=[2, 1, 0])
    
    print(f"  [OK] Created RGB composites (both at {after_width} × {after_height} pixels, 15m resolution)")
    print()
    
    # Step 3: Create AOI overview visualization
    print("Step 3: Creating AOI overview visualization...")
    print("  Using BEFORE image (30m) for overview")
    
    # Create RGB from original before image for overview
    before_rgb_original = create_rgb_composite(before_bands, band_indices=[2, 1, 0])
    before_rgb_normalized = normalize_rgb_for_display(before_rgb_original)
    
    # Calculate image extent using original before image transform
    left = before_transform[2]
    top = before_transform[5]
    right = left + before_width * before_transform[0]
    bottom = top + before_height * before_transform[4]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Display full before RGB image
    ax.imshow(before_rgb_normalized, extent=[left, right, bottom, top])
    
    # Draw AOI outlines using original before image coordinates
    for aoi_idx, aoi in enumerate(aoi_config):
        center_col, center_row = aoi['center']
        side_length = aoi['side_length']
        
        half_side = side_length // 2
        row_start = center_row - half_side
        row_end = center_row + half_side + (side_length % 2)
        col_start = center_col - half_side
        col_end = center_col + half_side + (side_length % 2)
        
        # Convert to map coordinates using original before image transform
        x_start, y_start = pixel_to_map(col_start, row_start, before_transform)
        x_end, y_end = pixel_to_map(col_end, row_end, before_transform)
        
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
    ax.set_title('Before Diffusion RGB Image (30m) with AOI Outlines', 
                 fontsize=14, fontweight='bold', pad=10, y=1.02)
    
    # Add legend
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=len(aoi_config), 
              frameon=False, fontsize=10)
    
    plt.tight_layout()
    overview_output_path = output_dir / 'diffusion_aoi_overview.png'
    plt.savefig(overview_output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {overview_output_path}")
    plt.close()
    print()
    
    # Step 4: Extract AOI regions
    print("Step 4: Extracting AOI regions...")
    print("  Converting AOI coordinates from before image (30m) to after image (15m) grid...")
    
    # Calculate scale factor: after image has 2x resolution
    scale_factor = abs(before_transform[0]) / abs(after_transform[0])
    print(f"  Scale factor (before->after): {scale_factor:.2f}x")
    
    # Now both images are aligned to the same grid (after image's 15m grid)
    aoi_rgb_regions = {
        'before': [],
        'after': []
    }
    
    for aoi in aoi_config:
        # Original AOI coordinates are in before image (30m) pixel space
        center_col_before, center_row_before = aoi['center']
        side_length_before = aoi['side_length']
        
        # Convert to after image (15m) pixel space
        center_col_after = int(center_col_before * scale_factor)
        center_row_after = int(center_row_before * scale_factor)
        side_length_after = int(side_length_before * scale_factor)
        
        # Extract from before RGB (already reprojected to after image grid)
        before_r_region, bounds = extract_square_region(
            before_rgb[:, :, 0], int(center_row_after), int(center_col_after), side_length_after
        )
        before_g_region, _ = extract_square_region(
            before_rgb[:, :, 1], int(center_row_after), int(center_col_after), side_length_after
        )
        before_b_region, _ = extract_square_region(
            before_rgb[:, :, 2], int(center_row_after), int(center_col_after), side_length_after
        )
        before_rgb_aoi = np.dstack([before_r_region, before_g_region, before_b_region])
        aoi_rgb_regions['before'].append(before_rgb_aoi)
        
        # Extract from after RGB (using same coordinates since images are aligned)
        after_r_region, _ = extract_square_region(
            after_rgb[:, :, 0], int(center_row_after), int(center_col_after), side_length_after
        )
        after_g_region, _ = extract_square_region(
            after_rgb[:, :, 1], int(center_row_after), int(center_col_after), side_length_after
        )
        after_b_region, _ = extract_square_region(
            after_rgb[:, :, 2], int(center_row_after), int(center_col_after), side_length_after
        )
        after_rgb_aoi = np.dstack([after_r_region, after_g_region, after_b_region])
        aoi_rgb_regions['after'].append(after_rgb_aoi)
        
        print(f"  [OK] Extracted {aoi['name']} (before: {side_length_before}px @ 30m -> {side_length_after}px @ 15m)")
    
    print()
    
    # Step 5: Normalize for display
    print("Step 5: Normalizing images for display...")
    
    # Normalize RGB composites (consistent scaling per row)
    # Collect all regions for each row to compute global min/max per channel
    before_all_r = []
    before_all_g = []
    before_all_b = []
    after_all_r = []
    after_all_g = []
    after_all_b = []
    
    for reg in aoi_rgb_regions['before']:
        before_all_r.extend(reg[:, :, 0].flatten())
        before_all_g.extend(reg[:, :, 1].flatten())
        before_all_b.extend(reg[:, :, 2].flatten())
    
    for reg in aoi_rgb_regions['after']:
        after_all_r.extend(reg[:, :, 0].flatten())
        after_all_g.extend(reg[:, :, 1].flatten())
        after_all_b.extend(reg[:, :, 2].flatten())
    
    # Calculate percentile ranges for consistent scaling per row
    before_percentiles = {
        'R': (np.percentile(before_all_r, 2), np.percentile(before_all_r, 98)),
        'G': (np.percentile(before_all_g, 2), np.percentile(before_all_g, 98)),
        'B': (np.percentile(before_all_b, 2), np.percentile(before_all_b, 98))
    }
    after_percentiles = {
        'R': (np.percentile(after_all_r, 2), np.percentile(after_all_r, 98)),
        'G': (np.percentile(after_all_g, 2), np.percentile(after_all_g, 98)),
        'B': (np.percentile(after_all_b, 2), np.percentile(after_all_b, 98))
    }
    
    # Calculate separate percentiles for AOI 2 (col_idx=1) to make it brighter
    # Use wider percentile range (0.5-99.5) to stretch values more
    aoi2_idx = 1  # AOI 2 is the 2nd column (0-indexed)
    before_aoi2_r = aoi_rgb_regions['before'][aoi2_idx][:, :, 0].flatten()
    before_aoi2_g = aoi_rgb_regions['before'][aoi2_idx][:, :, 1].flatten()
    before_aoi2_b = aoi_rgb_regions['before'][aoi2_idx][:, :, 2].flatten()
    after_aoi2_r = aoi_rgb_regions['after'][aoi2_idx][:, :, 0].flatten()
    after_aoi2_g = aoi_rgb_regions['after'][aoi2_idx][:, :, 1].flatten()
    after_aoi2_b = aoi_rgb_regions['after'][aoi2_idx][:, :, 2].flatten()
    
    before_percentiles_aoi2 = {
        'R': (np.percentile(before_aoi2_r, 0.5), np.percentile(before_aoi2_r, 99.5)),
        'G': (np.percentile(before_aoi2_g, 0.5), np.percentile(before_aoi2_g, 99.5)),
        'B': (np.percentile(before_aoi2_b, 0.5), np.percentile(before_aoi2_b, 99.5))
    }
    after_percentiles_aoi2 = {
        'R': (np.percentile(after_aoi2_r, 0.5), np.percentile(after_aoi2_r, 99.5)),
        'G': (np.percentile(after_aoi2_g, 0.5), np.percentile(after_aoi2_g, 99.5)),
        'B': (np.percentile(after_aoi2_b, 0.5), np.percentile(after_aoi2_b, 99.5))
    }
    
    # Print diagnostic info about data ranges
    print(f"  Before image data ranges (2-98 percentile):")
    print(f"    R: {before_percentiles['R'][0]:.4f} to {before_percentiles['R'][1]:.4f}")
    print(f"    G: {before_percentiles['G'][0]:.4f} to {before_percentiles['G'][1]:.4f}")
    print(f"    B: {before_percentiles['B'][0]:.4f} to {before_percentiles['B'][1]:.4f}")
    print(f"  After image data ranges (2-98 percentile):")
    print(f"    R: {after_percentiles['R'][0]:.4f} to {after_percentiles['R'][1]:.4f}")
    print(f"    G: {after_percentiles['G'][0]:.4f} to {after_percentiles['G'][1]:.4f}")
    print(f"    B: {after_percentiles['B'][0]:.4f} to {after_percentiles['B'][1]:.4f}")
    print(f"  [OK] Normalization ranges computed")
    print()
    
    # Step 6: Create 2xN grid visualization (rotated: rows = before/after, columns = AOIs)
    print("Step 6: Creating 2xN comparison grid...")
    
    num_aois = len(aoi_config)
    fig, axes = plt.subplots(2, num_aois, figsize=(4*num_aois, 8))
    fig.suptitle('Diffusion Comparison: Before vs After', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Row 0: Before Diffusion
    for col_idx, aoi in enumerate(aoi_config):
        ax = axes[0, col_idx]
        before_rgb_aoi = aoi_rgb_regions['before'][col_idx]
        before_normalized = np.zeros_like(before_rgb_aoi, dtype=np.uint8)
        
        # Use brighter normalization for AOI 2 (col_idx=1)
        if col_idx == 1:
            percentiles_to_use = before_percentiles_aoi2
        else:
            percentiles_to_use = before_percentiles
        
        for ch_idx, color in enumerate(['R', 'G', 'B']):
            channel = before_rgb_aoi[:, :, ch_idx]
            min_val, max_val = percentiles_to_use[color]
            if max_val > min_val:
                normalized_ch = np.clip((channel - min_val) / (max_val - min_val) * 255, 0, 255)
                before_normalized[:, :, ch_idx] = normalized_ch.astype(np.uint8)
        ax.imshow(before_normalized, aspect='equal', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(aoi_colors[col_idx])
            spine.set_linewidth(3)
        if col_idx == 0:
            ax.set_ylabel('Before Diffusion', fontsize=12, fontweight='bold', rotation=90, labelpad=20)
        ax.set_title(aoi['name'], fontsize=12, fontweight='bold', pad=10)
    
    # Row 1: After Diffusion
    for col_idx, aoi in enumerate(aoi_config):
        ax = axes[1, col_idx]
        after_rgb_aoi = aoi_rgb_regions['after'][col_idx]
        after_normalized = np.zeros_like(after_rgb_aoi, dtype=np.uint8)
        
        # Use brighter normalization for AOI 2 (col_idx=1)
        if col_idx == 1:
            percentiles_to_use = after_percentiles_aoi2
        else:
            percentiles_to_use = after_percentiles
        
        for ch_idx, color in enumerate(['R', 'G', 'B']):
            channel = after_rgb_aoi[:, :, ch_idx]
            min_val, max_val = percentiles_to_use[color]
            if max_val > min_val:
                normalized_ch = np.clip((channel - min_val) / (max_val - min_val) * 255, 0, 255)
                after_normalized[:, :, ch_idx] = normalized_ch.astype(np.uint8)
        ax.imshow(after_normalized, aspect='equal', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(aoi_colors[col_idx])
            spine.set_linewidth(3)
        if col_idx == 0:
            ax.set_ylabel('After Diffusion', fontsize=12, fontweight='bold', rotation=90, labelpad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'diffusion_comparison_2xN.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()
    print()
    
    # Final summary
    print("=" * 80)
    print("DIFFUSION COMPARISON VISUALIZATION SUMMARY")
    print("=" * 80)
    print(f"\nImages compared:")
    print(f"  - Before: {before_width} × {before_height} pixels (30m, reprojected to 15m grid)")
    print(f"  - After: {after_width} × {after_height} pixels (15m native resolution)")
    
    print(f"\nAOIs analyzed: {num_aois}")
    for aoi in aoi_config:
        print(f"  - {aoi['name']}: center=({aoi['center'][0]}, {aoi['center'][1]}), size={aoi['side_length']}px")
    
    print(f"\nOutput files:")
    print(f"  - AOI overview: diffusion_aoi_overview.png")
    print(f"  - Comparison grid: diffusion_comparison_2xN.png")
    
    print("\n" + "=" * 80)
    print("[OK] Visualization completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

