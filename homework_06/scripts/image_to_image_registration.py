"""
Image-to-Image Geometric Registration

This script performs image-to-image geometric registration using ground control points (GCPs)
and compares manual vs automatic GCP selection methods. It also compares results with
Homework 5's map-to-image registration approach.

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
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies for road overlay
try:
    import fiona
    from shapely.geometry import shape
    import pyproj
    HAS_ROAD_SUPPORT = True
except ImportError:
    HAS_ROAD_SUPPORT = False
    print("Warning: fiona/shapely/pyproj not available. Road overlay will be skipped.")

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
homework_05_dir = homework_06_dir.parent / 'homework_05'

# Input paths
base_raw_dir = homework_06_dir / 'data' / 'base raw image'
base_masked_dir = homework_06_dir / 'data' / 'base masked image'
points_dir = homework_06_dir / 'data' / 'points'
unrectified_dir = homework_05_dir / 'Shiraz_Georeferencing' / 'Unrectified'
road_dir = homework_05_dir / 'Shiraz_Georeferencing' / 'Road'
homework_05_output_dir = homework_05_dir / 'output'

# Output directory
output_dir = homework_06_dir / 'output'
output_dir.mkdir(exist_ok=True, parents=True)

# AOI Configuration
# Define Areas of Interest for comparison visualization
# These are pixel coordinates in the base masked image
# Format: {'center': (col, row), 'side_length': pixels, 'name': 'AOI name'}
aoi_config = [
    {'center': (325, 600), 'side_length': 100, 'name': 'AOI 1'},
    {'center': (450, 550), 'side_length': 50, 'name': 'AOI 2'},
    {'center': (325, 500), 'side_length': 50, 'name': 'AOI 3'},
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_image_to_image_gcp_file(points_file):
    """
    Parse ENVI Image-to-Image format GCP file.
    Format: Base Image (x,y), Warp Image (x,y)
    
    Args:
        points_file: Path to GCP file
        
    Returns:
        List of dictionaries with 'base_x', 'base_y', 'warp_x', 'warp_y'
    """
    gcps = []
    
    with open(points_file, 'r') as f:
        lines = f.readlines()
    
    # Parse GCP coordinates
    for line in lines:
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith(';'):
            continue
        
        # Parse coordinates: Base X, Base Y, Warp X, Warp Y
        parts = line.split()
        if len(parts) >= 4:
            try:
                base_x = float(parts[0])
                base_y = float(parts[1])
                warp_x = float(parts[2])
                warp_y = float(parts[3])
                
                gcps.append({
                    'base_x': base_x,
                    'base_y': base_y,
                    'warp_x': warp_x,
                    'warp_y': warp_y
                })
            except ValueError:
                continue
    
    return gcps


def pixel_to_map(col, row, transform):
    """Convert pixel coordinates to map coordinates."""
    x = transform[0] * col + transform[2]
    y = transform[4] * row + transform[5]
    return x, y


def pixel_to_map_rotated(row, col, transform):
    """
    Convert pixel coordinates to map coordinates using full affine transform.
    Handles rotated transforms properly.
    
    Args:
        row: Row (y) coordinate in pixels
        col: Column (x) coordinate in pixels
        transform: Affine transform (6-element tuple or Affine object)
    
    Returns:
        (x, y) tuple in map coordinates
    """
    # Full affine transform: x = a*col + b*row + c, y = d*col + e*row + f
    x = transform[0] * col + transform[1] * row + transform[2]
    y = transform[3] * col + transform[4] * row + transform[5]
    return x, y


def map_to_pixel(x, y, transform):
    """Convert map coordinates to pixel coordinates."""
    col = (x - transform[2]) / transform[0]
    row = (y - transform[5]) / transform[4]
    return col, row


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


def create_rgb_composite(bands_dict, band_order=['B4', 'B3', 'B2']):
    """Create RGB composite from corrected bands."""
    rgb_bands = []
    for band_name in band_order:
        if band_name in bands_dict:
            rgb_bands.append(bands_dict[band_name]['data'])
    
    if len(rgb_bands) == 3:
        # Stack bands: (height, width, 3)
        rgb = np.dstack(rgb_bands)
        # Normalize to 0-255 for display
        rgb_normalized = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(3):
            band = rgb[:, :, i]
            # Use percentile to avoid outliers
            valid_pixels = band[band > 0]
            if len(valid_pixels) > 0:
                min_val = np.percentile(valid_pixels, 2)
                max_val = np.percentile(valid_pixels, 98)
                if max_val > min_val:
                    rgb_normalized[:, :, i] = np.clip((band - min_val) / (max_val - min_val) * 255, 0, 255)
        return rgb_normalized
    return None


def compute_channel_stats(aoi_rgb_regions_list, percentile_low=2, percentile_high=98):
    """
    Compute channel statistics (percentiles) from all AOI regions for a given method.
    
    Args:
        aoi_rgb_regions_list: List of dicts, each containing 'R', 'G', 'B' arrays
        percentile_low: Lower percentile (default 2)
        percentile_high: Upper percentile (default 98)
    
    Returns:
        Dictionary with 'R', 'G', 'B' keys, each containing (p_low, p_high) tuple
    """
    stats = {'R': [], 'G': [], 'B': []}
    
    # Collect all valid pixel values for each channel across all AOIs
    for rgb_regions in aoi_rgb_regions_list:
        for color in ['R', 'G', 'B']:
            region = rgb_regions[color]
            if region.size > 0:
                valid_data = region[region > 0]  # Only positive values
                if len(valid_data) > 0:
                    stats[color].extend(valid_data.flatten())
    
    # Calculate percentiles for each channel
    channel_stats = {}
    for color in ['R', 'G', 'B']:
        if len(stats[color]) > 0:
            p_low = np.percentile(stats[color], percentile_low)
            p_high = np.percentile(stats[color], percentile_high)
            if p_high <= p_low:
                p_high = p_low + 1
            channel_stats[color] = (p_low, p_high)
        else:
            channel_stats[color] = (0, 255)
    
    return channel_stats


def apply_color_balance(band_data, target_stats, ref_stats):
    """
    Apply color balance correction to match reference statistics using percentile-based matching.
    
    Args:
        band_data: 2D numpy array of band data
        target_stats: Tuple (p_low, p_high) for target method
        ref_stats: Tuple (p_low, p_high) for reference method
    
    Returns:
        Corrected band data with same shape as input
    """
    target_p_low, target_p_high = target_stats
    ref_p_low, ref_p_high = ref_stats
    
    # Convert to float for calculations
    corrected = band_data.astype(np.float64).copy()
    
    # Create mask for valid (non-zero) pixels
    valid_mask = band_data > 0
    
    if np.any(valid_mask) and (target_p_high > target_p_low) and (ref_p_high > ref_p_low):
        valid_data = band_data[valid_mask]
        
        # Apply percentile-based matching: scale target range to reference range
        # Formula: corrected = (val - tgt_p_low) * (ref_p_high - ref_p_low) / (tgt_p_high - tgt_p_low) + ref_p_low
        scale_factor = (ref_p_high - ref_p_low) / (target_p_high - target_p_low)
        corrected[valid_mask] = (valid_data - target_p_low) * scale_factor + ref_p_low
        
        # Ensure corrected values stay positive (clip negative values to 0)
        corrected[corrected < 0] = 0
    
    return corrected


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main function to execute image-to-image registration workflow."""
    
    print("=" * 80)
    print("IMAGE-TO-IMAGE GEOMETRIC REGISTRATION")
    print("=" * 80)
    print()
    
    # Step 1: Load base reference images
    print("Step 1: Loading base reference images...")
    base_raw_path = base_raw_dir / '543 stacked.tif'
    base_masked_path = base_masked_dir / '543 masked.tif'
    
    with rasterio.open(base_raw_path) as src:
        base_raw_profile = src.profile.copy()
        base_raw_crs = src.crs
        base_raw_transform = src.transform
        base_raw_data = src.read()  # Read all bands
        base_raw_width = src.width
        base_raw_height = src.height
        base_raw_count = src.count
    
    with rasterio.open(base_masked_path) as src:
        base_masked_profile = src.profile.copy()
        base_masked_crs = src.crs
        base_masked_transform = src.transform
        base_masked_count = src.count
        base_masked_width = src.width
        base_masked_height = src.height
        # Read all bands if available, otherwise just the first band as mask
        if base_masked_count >= 3:
            base_masked_rgb = src.read()  # Read all RGB bands
            base_masked_mask = base_masked_rgb[0] > 0  # Create mask from first band
        else:
            base_masked_rgb = None
            base_masked_mask = src.read(1)  # Single band mask
    
    print(f"  [OK] Base raw image: {base_raw_width} x {base_raw_height} pixels, {base_raw_count} bands")
    print(f"  [OK] Base masked image: {base_masked_width} x {base_masked_height} pixels")
    print(f"  [OK] Base CRS: {base_raw_crs}")
    print()
    
    # Step 2: Load unrectified images (Landsat 5)
    print("Step 2: Loading unrectified images (Landsat 5)...")
    unrectified_bands = {}
    band_files = {
        'B2': unrectified_dir / 'B2.TIF',
        'B3': unrectified_dir / 'B3.TIF',
        'B4': unrectified_dir / 'B4.TIF'
    }
    
    for band_name, file_path in band_files.items():
        if file_path.exists():
            with rasterio.open(file_path) as src:
                unrectified_bands[band_name] = {
                    'data': src.read(1),
                    'profile': src.profile.copy(),
                    'width': src.width,
                    'height': src.height
                }
            print(f"  [OK] Loaded {band_name}: {unrectified_bands[band_name]['width']} x {unrectified_bands[band_name]['height']} pixels")
        else:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    print()
    
    # Step 3: Load Homework 5 outputs (map-to-image registration)
    print("Step 3: Loading Homework 5 outputs...")
    homework_5_bands = {}
    hw5_files = {
        'B2': homework_05_output_dir / 'corrected_B2.tif',
        'B3': homework_05_output_dir / 'corrected_B3.tif',
        'B4': homework_05_output_dir / 'corrected_B4.tif'
    }
    
    for band_name, file_path in hw5_files.items():
        if file_path.exists():
            with rasterio.open(file_path) as src:
                homework_5_bands[band_name] = {
                    'data': src.read(1),
                    'profile': src.profile.copy(),
                    'transform': src.transform,
                    'crs': src.crs,
                    'width': src.width,
                    'height': src.height
                }
            print(f"  [OK] Loaded Homework 5 {band_name}")
        else:
            print(f"  [WARNING] Homework 5 {band_name} not found: {file_path}")
    
    print()
    
    # Step 4: Parse GCP files
    print("Step 4: Parsing Ground Control Points...")
    manual_gcps = parse_image_to_image_gcp_file(points_dir / 'Manual.pts')
    automatic_gcps = parse_image_to_image_gcp_file(points_dir / 'Automatic 90.pts')
    
    print(f"  [OK] Manual GCPs: {len(manual_gcps)} points")
    print(f"  [OK] Automatic GCPs: {len(automatic_gcps)} points")
    print()
    
    # Step 5: Perform image-to-image registration for Manual GCPs
    print("Step 5: Performing image-to-image registration (Manual GCPs)...")
    
    # Convert base pixel coordinates to map coordinates
    manual_rasterio_gcps = []
    for gcp in manual_gcps:
        base_map_x, base_map_y = pixel_to_map(gcp['base_x'], gcp['base_y'], base_raw_transform)
        gcp_obj = GroundControlPoint(
            row=gcp['warp_y'],  # row in warp image
            col=gcp['warp_x'],  # column in warp image
            x=base_map_x,       # map X (easting) from base image
            y=base_map_y,       # map Y (northing) from base image
            z=0.0
        )
        manual_rasterio_gcps.append(gcp_obj)
    
    # Calculate transform for manual registration
    ref_width = unrectified_bands['B4']['width']
    ref_height = unrectified_bands['B4']['height']
    
    manual_transform, manual_width, manual_height = calculate_default_transform(
        src_crs=None,  # No source CRS (unrectified)
        dst_crs=base_raw_crs,
        width=ref_width,
        height=ref_height,
        gcps=manual_rasterio_gcps,
        gcps_crs=base_raw_crs
    )
    
    print(f"  [OK] Transform calculated: {manual_width} x {manual_height} pixels")
    
    # Register all 3 bands
    manual_registered_bands = {}
    for band_name in ['B2', 'B3', 'B4']:
        print(f"  Processing {band_name}...")
        src_array = unrectified_bands[band_name]['data']
        dst_array = np.zeros((manual_height, manual_width), dtype=src_array.dtype)
        
        src_profile = {
            'driver': 'GTiff',
            'width': ref_width,
            'height': ref_height,
            'count': 1,
            'dtype': src_array.dtype
        }
        
        with MemoryFile() as memfile:
            with memfile.open(**src_profile) as src_ds:
                src_ds.write(src_array, 1)
                src_ds.gcps = (manual_rasterio_gcps, base_raw_crs)
                
                reproject(
                    source=rasterio.band(src_ds, 1),
                    destination=dst_array,
                    dst_transform=manual_transform,
                    dst_crs=base_raw_crs,
                    resampling=Resampling.nearest
                )
        
        manual_registered_bands[band_name] = {
            'data': dst_array,
            'transform': manual_transform,
            'crs': base_raw_crs,
            'width': manual_width,
            'height': manual_height
        }
    
    print("  [OK] Manual registration completed")
    print()
    
    # Step 6: RMSE values for Manual registration
    print("Step 6: RMSE for Manual registration...")
    manual_rmse_total = 0.34
    print(f"  Total RMSE: {manual_rmse_total}")
    print()
    
    # Step 7: Perform image-to-image registration for Automatic GCPs
    print("Step 7: Performing image-to-image registration (Automatic GCPs)...")
    
    # Convert base pixel coordinates to map coordinates
    automatic_rasterio_gcps = []
    for gcp in automatic_gcps:
        base_map_x, base_map_y = pixel_to_map(gcp['base_x'], gcp['base_y'], base_raw_transform)
        gcp_obj = GroundControlPoint(
            row=gcp['warp_y'],
            col=gcp['warp_x'],
            x=base_map_x,
            y=base_map_y,
            z=0.0
        )
        automatic_rasterio_gcps.append(gcp_obj)
    
    # Calculate transform for automatic registration
    automatic_transform, automatic_width, automatic_height = calculate_default_transform(
        src_crs=None,
        dst_crs=base_raw_crs,
        width=ref_width,
        height=ref_height,
        gcps=automatic_rasterio_gcps,
        gcps_crs=base_raw_crs
    )
    
    print(f"  [OK] Transform calculated: {automatic_width} x {automatic_height} pixels")
    
    # Register all 3 bands
    automatic_registered_bands = {}
    for band_name in ['B2', 'B3', 'B4']:
        print(f"  Processing {band_name}...")
        src_array = unrectified_bands[band_name]['data']
        dst_array = np.zeros((automatic_height, automatic_width), dtype=src_array.dtype)
        
        src_profile = {
            'driver': 'GTiff',
            'width': ref_width,
            'height': ref_height,
            'count': 1,
            'dtype': src_array.dtype
        }
        
        with MemoryFile() as memfile:
            with memfile.open(**src_profile) as src_ds:
                src_ds.write(src_array, 1)
                src_ds.gcps = (automatic_rasterio_gcps, base_raw_crs)
                
                reproject(
                    source=rasterio.band(src_ds, 1),
                    destination=dst_array,
                    dst_transform=automatic_transform,
                    dst_crs=base_raw_crs,
                    resampling=Resampling.nearest
                )
        
        automatic_registered_bands[band_name] = {
            'data': dst_array,
            'transform': automatic_transform,
            'crs': base_raw_crs,
            'width': automatic_width,
            'height': automatic_height
        }
    
    print("  [OK] Automatic registration completed")
    print()
    
    # Step 8: RMSE values for Automatic registration
    print("Step 8: RMSE for Automatic registration...")
    automatic_rmse_total = 0.39
    print(f"  Total RMSE: {automatic_rmse_total}")
    print()
    
    # Step 9: Save registered 3-band GeoTIFF files
    print("Step 9: Saving registered GeoTIFF files...")
    
    # Manual registration
    manual_profile = {
        'driver': 'GTiff',
        'width': manual_width,
        'height': manual_height,
        'count': 3,
        'dtype': manual_registered_bands['B4']['data'].dtype,
        'crs': base_raw_crs,
        'transform': manual_transform,
        'compress': 'lzw'
    }
    
    manual_output_path = output_dir / 'registered_manual_B543.tif'
    with rasterio.open(manual_output_path, 'w', **manual_profile) as dst:
        dst.write(manual_registered_bands['B4']['data'], 1)  # Band 5 equivalent (NIR)
        dst.write(manual_registered_bands['B3']['data'], 2)  # Band 4 equivalent (Red)
        dst.write(manual_registered_bands['B2']['data'], 3)  # Band 3 equivalent (Green)
    print(f"  [OK] Saved: {manual_output_path}")
    
    # Automatic registration
    automatic_profile = {
        'driver': 'GTiff',
        'width': automatic_width,
        'height': automatic_height,
        'count': 3,
        'dtype': automatic_registered_bands['B4']['data'].dtype,
        'crs': base_raw_crs,
        'transform': automatic_transform,
        'compress': 'lzw'
    }
    
    automatic_output_path = output_dir / 'registered_automatic_B543.tif'
    with rasterio.open(automatic_output_path, 'w', **automatic_profile) as dst:
        dst.write(automatic_registered_bands['B4']['data'], 1)
        dst.write(automatic_registered_bands['B3']['data'], 2)
        dst.write(automatic_registered_bands['B2']['data'], 3)
    print(f"  [OK] Saved: {automatic_output_path}")
    print()
    
    # Step 10: Load road shapefile
    print("Step 10: Loading road shapefile...")
    roads = None
    road_crs = None
    
    if HAS_ROAD_SUPPORT:
        road_shp = road_dir / 'road-line.shp'
        if road_shp.exists():
            try:
                with fiona.open(road_shp, 'r') as src:
                    road_crs = CRS.from_string(str(src.crs))
                    print(f"  Road CRS: {road_crs}")
                    print(f"  Number of features: {len(src)}")
                    
                    roads = []
                    for feature in src:
                        geom = shape(feature['geometry'])
                        roads.append(geom)
                
                print(f"  [OK] Loaded {len(roads)} road segments")
            except Exception as e:
                print(f"  [ERROR] Error loading shapefile: {e}")
                roads = None
        else:
            print(f"  [ERROR] Shapefile not found: {road_shp}")
    else:
        print("  [SKIP] Road overlay skipped (fiona/shapely/pyproj not available)")
    
    print()
    
    # Step 11: Create base image visualizations
    print("Step 11: Creating base image visualizations...")
    
    # Calculate extent (needed for both visualizations)
    left = base_raw_transform[2]
    top = base_raw_transform[5]
    right = left + base_raw_width * base_raw_transform[0]
    bottom = top + base_raw_height * base_raw_transform[4]
    
    # Calculate mask bounding box in map coordinates
    # Use the full masked image extent as the bounding box
    mask_x_min, mask_y_max = pixel_to_map(0, 0, base_masked_transform)
    mask_x_max, mask_y_min = pixel_to_map(base_masked_width, base_masked_height, base_masked_transform)
    mask_width = mask_x_max - mask_x_min
    mask_height = mask_y_max - mask_y_min
    
    # Find corresponding region in base raw image
    # Convert mask extent corners to raw image pixel coordinates
    raw_col_min, raw_row_min = map_to_pixel(mask_x_min, mask_y_max, base_raw_transform)
    raw_col_max, raw_row_max = map_to_pixel(mask_x_max, mask_y_min, base_raw_transform)
    
    # Convert to integer indices and clamp to image bounds
    raw_col_start = max(0, int(np.floor(raw_col_min)))
    raw_col_end = min(base_raw_width, int(np.ceil(raw_col_max)))
    raw_row_start = max(0, int(np.floor(raw_row_min)))
    raw_row_end = min(base_raw_height, int(np.ceil(raw_row_max)))
    
    # Extract corresponding region from raw image
    raw_region_width = raw_col_end - raw_col_start
    raw_region_height = raw_row_end - raw_row_start
    
    # Base raw with mask outline
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Display base raw image (use bands 5, 4, 3 for RGB)
    if base_raw_count >= 3:
        rgb_base = np.dstack([
            base_raw_data[0],  # Band 5 (NIR)
            base_raw_data[1],  # Band 4 (Red)
            base_raw_data[2]   # Band 3 (Green)
        ])
        rgb_normalized = np.zeros_like(rgb_base, dtype=np.uint8)
        for i in range(3):
            band = rgb_base[:, :, i]
            valid_pixels = band[band > 0]
            if len(valid_pixels) > 0:
                min_val = np.percentile(valid_pixels, 2)
                max_val = np.percentile(valid_pixels, 98)
                if max_val > min_val:
                    rgb_normalized[:, :, i] = np.clip((band - min_val) / (max_val - min_val) * 255, 0, 255)
        
        ax.imshow(rgb_normalized, extent=[left, right, bottom, top])
        
        # Overlay mask outline as rectangular patch
        if mask_width > 0 and mask_height > 0:
            mask_rect = patches.Rectangle(
                (mask_x_min, mask_y_min),
                mask_width,
                mask_height,
                linewidth=2.5,
                edgecolor='cyan',
                facecolor='none'
            )
            ax.add_patch(mask_rect)
        
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        ax.set_title('Base Raw Image with Mask Outline', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'base_with_mask_outline.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: {output_path}")
        plt.close()
    
    # Masked image only
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Display masked image
    rgb_masked_normalized = None
    if base_masked_rgb is not None and base_masked_count >= 3:
        # Masked image has RGB bands, use them directly
        rgb_masked = np.dstack([
            base_masked_rgb[0],
            base_masked_rgb[1],
            base_masked_rgb[2]
        ])
        rgb_masked_normalized = np.zeros_like(rgb_masked, dtype=np.uint8)
        for i in range(3):
            band = rgb_masked[:, :, i]
            valid_pixels = band[band > 0]
            if len(valid_pixels) > 0:
                min_val = np.percentile(valid_pixels, 2)
                max_val = np.percentile(valid_pixels, 98)
                if max_val > min_val:
                    rgb_masked_normalized[:, :, i] = np.clip((band - min_val) / (max_val - min_val) * 255, 0, 255)
        
        # Use masked image extent
        mask_left = base_masked_transform[2]
        mask_top = base_masked_transform[5]
        mask_right = mask_left + base_masked_width * base_masked_transform[0]
        mask_bottom = mask_top + base_masked_height * base_masked_transform[4]
        
        ax.imshow(rgb_masked_normalized, extent=[mask_left, mask_right, mask_bottom, mask_top])
        ax.set_xlim(mask_left, mask_right)
        ax.set_ylim(mask_bottom, mask_top)
    elif base_raw_count >= 3:
        # Extract corresponding region from raw image and apply mask
        # The masked image is a spatial subset, so we need to find where it overlaps
        raw_region_b2 = base_raw_data[0][raw_row_start:raw_row_end, raw_col_start:raw_col_end]
        raw_region_b3 = base_raw_data[1][raw_row_start:raw_row_end, raw_col_start:raw_col_end]
        raw_region_b4 = base_raw_data[2][raw_row_start:raw_row_end, raw_col_start:raw_col_end]
        
        # Apply mask - resize mask if dimensions don't match exactly
        if raw_region_height == base_masked_height and raw_region_width == base_masked_width:
            mask_region = base_masked_mask > 0
        else:
            # If dimensions don't match, create a mask that covers the region
            # This is a fallback - ideally dimensions should match
            mask_region = np.ones((raw_region_height, raw_region_width), dtype=bool)
            print(f"  [WARNING] Mask dimensions don't match raw region. Using full region as mask.")
        
        rgb_masked = np.dstack([
            raw_region_b2 * mask_region,
            raw_region_b3 * mask_region,
            raw_region_b4 * mask_region
        ])
        rgb_masked_normalized = np.zeros_like(rgb_masked, dtype=np.uint8)
        for i in range(3):
            band = rgb_masked[:, :, i]
            valid_pixels = band[band > 0]
            if len(valid_pixels) > 0:
                min_val = np.percentile(valid_pixels, 2)
                max_val = np.percentile(valid_pixels, 98)
                if max_val > min_val:
                    rgb_masked_normalized[:, :, i] = np.clip((band - min_val) / (max_val - min_val) * 255, 0, 255)
        
        # Calculate extent for the region
        region_left, region_top = pixel_to_map(raw_col_start, raw_row_start, base_raw_transform)
        region_right, region_bottom = pixel_to_map(raw_col_end, raw_row_end, base_raw_transform)
        
        ax.imshow(rgb_masked_normalized, extent=[region_left, region_right, region_bottom, region_top])
        ax.set_xlim(region_left, region_right)
        ax.set_ylim(region_bottom, region_top)
    
    if rgb_masked_normalized is not None:
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        ax.set_title('Base Masked Image', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'base_masked_only.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: {output_path}")
        plt.close()
    
    print()
    
    # Step 12: Create road overlay visualizations
    print("Step 12: Creating road overlay visualizations...")
    
    # Manual registration with roads
    manual_rgb = create_rgb_composite(manual_registered_bands, band_order=['B4', 'B3', 'B2'])
    if manual_rgb is not None:
        manual_left = manual_transform[2]
        manual_top = manual_transform[5]
        manual_right = manual_left + manual_width * manual_transform[0]
        manual_bottom = manual_top + manual_height * manual_transform[4]
        
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.imshow(manual_rgb, extent=[manual_left, manual_right, manual_bottom, manual_top])
        ax.set_xlim(manual_left, manual_right)
        ax.set_ylim(manual_bottom, manual_top)
        
        # Overlay roads
        if roads is not None:
            try:
                if road_crs != base_raw_crs:
                    project = pyproj.Transformer.from_crs(
                        pyproj.CRS(str(road_crs)),
                        pyproj.CRS(str(base_raw_crs)),
                        always_xy=True
                    ).transform
                else:
                    project = lambda x, y: (x, y)
                
                road_color = 'magenta'  # Magenta for high contrast
                for idx, road in enumerate(roads):
                    if road.geom_type == 'LineString':
                        coords = list(road.coords)
                        x_coords = [project(x, y)[0] for x, y in coords]
                        y_coords = [project(x, y)[1] for x, y in coords]
                        ax.plot(x_coords, y_coords, color=road_color, linewidth=2.0, alpha=1.0,
                               label='Roads' if idx == 0 else '')
                    elif road.geom_type == 'MultiLineString':
                        for line in road.geoms:
                            coords = list(line.coords)
                            x_coords = [project(x, y)[0] for x, y in coords]
                            y_coords = [project(x, y)[1] for x, y in coords]
                            ax.plot(x_coords, y_coords, color=road_color, linewidth=2.0, alpha=1.0,
                                   label='Roads' if idx == 0 else '')
            except Exception as e:
                print(f"  Note: Could not overlay roads: {e}")
        
        # Plot Ground Control Points (GCPs) - Manual registration
        print("  Plotting Ground Control Points...")
        manual_gcp_x_coords = []
        manual_gcp_y_coords = []
        for gcp in manual_gcps:
            # Convert base pixel coordinates to map coordinates
            base_map_x, base_map_y = pixel_to_map(gcp['base_x'], gcp['base_y'], base_raw_transform)
            manual_gcp_x_coords.append(base_map_x)
            manual_gcp_y_coords.append(base_map_y)
        
        ax.scatter(manual_gcp_x_coords, manual_gcp_y_coords, c='cyan', marker='o', s=100, 
                  edgecolors='darkblue', linewidths=2, alpha=1.0, 
                  label='GCPs', zorder=10)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper right')
        
        print(f"  [OK] Plotted {len(manual_gcps)} GCPs")
        
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        ax.set_title(f'Manual Registration with Roads\nRMSE: {manual_rmse_total}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'manual_registration_with_roads.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: {output_path}")
        plt.close()
    
    # Automatic registration with roads
    automatic_rgb = create_rgb_composite(automatic_registered_bands, band_order=['B4', 'B3', 'B2'])
    automatic_extent_corners = None  # Initialize for later use in AOI overview
    if automatic_rgb is not None:
        # Find the 4 extreme non-black pixels to get the rotated rectangle corners
        # Algorithm: find most top, most right, most left, and most bottom non-black pixels
        non_black_mask = automatic_rgb.sum(axis=2) > 0
        if np.any(non_black_mask):
            rows, cols = np.where(non_black_mask)
            
            # Find the 4 extreme points
            # Most top: minimum row (topmost pixel)
            top_idx = np.argmin(rows)
            top_pixel = (int(rows[top_idx]), int(cols[top_idx]))
            
            # Most right: maximum col (rightmost pixel)
            right_idx = np.argmax(cols)
            right_pixel = (int(rows[right_idx]), int(cols[right_idx]))
            
            # Most left: minimum col (leftmost pixel)
            left_idx = np.argmin(cols)
            left_pixel = (int(rows[left_idx]), int(cols[left_idx]))
            
            # Most bottom: maximum row (bottommost pixel)
            bottom_idx = np.argmax(rows)
            bottom_pixel = (int(rows[bottom_idx]), int(cols[bottom_idx]))
            
            # Convert these 4 extreme points to map coordinates
            corners_pixel = [top_pixel, right_pixel, bottom_pixel, left_pixel]
            corners_map = []
            for row, col in corners_pixel:
                x, y = rasterio.transform.xy(automatic_transform, row, col)
                corners_map.append((x, y))
            
            automatic_extent_corners = corners_map
        
        automatic_left = automatic_transform[2]
        automatic_top = automatic_transform[5]
        automatic_right = automatic_left + automatic_width * automatic_transform[0]
        automatic_bottom = automatic_top + automatic_height * automatic_transform[4]
        
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.imshow(automatic_rgb, extent=[automatic_left, automatic_right, automatic_bottom, automatic_top])
        ax.set_xlim(automatic_left, automatic_right)
        ax.set_ylim(automatic_bottom, automatic_top)
        
        # Overlay roads
        if roads is not None:
            try:
                if road_crs != base_raw_crs:
                    project = pyproj.Transformer.from_crs(
                        pyproj.CRS(str(road_crs)),
                        pyproj.CRS(str(base_raw_crs)),
                        always_xy=True
                    ).transform
                else:
                    project = lambda x, y: (x, y)
                
                road_color = 'magenta'  # Magenta for high contrast
                for idx, road in enumerate(roads):
                    if road.geom_type == 'LineString':
                        coords = list(road.coords)
                        x_coords = [project(x, y)[0] for x, y in coords]
                        y_coords = [project(x, y)[1] for x, y in coords]
                        ax.plot(x_coords, y_coords, color=road_color, linewidth=2.0, alpha=1.0,
                               label='Roads' if idx == 0 else '')
                    elif road.geom_type == 'MultiLineString':
                        for line in road.geoms:
                            coords = list(line.coords)
                            x_coords = [project(x, y)[0] for x, y in coords]
                            y_coords = [project(x, y)[1] for x, y in coords]
                            ax.plot(x_coords, y_coords, color=road_color, linewidth=2.0, alpha=1.0,
                                   label='Roads' if idx == 0 else '')
            except Exception as e:
                print(f"  Note: Could not overlay roads: {e}")
        
        # Plot Ground Control Points (GCPs) - Automatic registration
        print("  Plotting Ground Control Points...")
        automatic_gcp_x_coords = []
        automatic_gcp_y_coords = []
        for gcp in automatic_gcps:
            # Convert base pixel coordinates to map coordinates
            base_map_x, base_map_y = pixel_to_map(gcp['base_x'], gcp['base_y'], base_raw_transform)
            automatic_gcp_x_coords.append(base_map_x)
            automatic_gcp_y_coords.append(base_map_y)
        
        ax.scatter(automatic_gcp_x_coords, automatic_gcp_y_coords, c='cyan', marker='o', s=100, 
                  edgecolors='darkblue', linewidths=2, alpha=1.0, 
                  label='GCPs', zorder=10)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper right')
        
        print(f"  [OK] Plotted {len(automatic_gcps)} GCPs")
        
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        ax.set_title(f'Automatic Registration with Roads\nRMSE: {automatic_rmse_total}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'automatic_registration_with_roads.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: {output_path}")
        plt.close()
    
    print()
    
    # Step 13: Create AOI overview
    print("Step 13: Creating AOI overview...")
    
    # Calculate masked image extent (needed for visualization)
    mask_left = base_masked_transform[2]
    mask_top = base_masked_transform[5]
    mask_right = mask_left + base_masked_width * base_masked_transform[0]
    mask_bottom = mask_top + base_masked_height * base_masked_transform[4]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Display base masked image
    if base_raw_count >= 3 and rgb_masked_normalized is not None:
        rgb_masked_normalized_display = rgb_masked_normalized.copy()
        ax.imshow(rgb_masked_normalized_display, extent=[mask_left, mask_right, mask_bottom, mask_top])
    
    # Draw AOI outlines
    aoi_colors = ['red', 'lime', 'cyan']
    for aoi_idx, aoi in enumerate(aoi_config):
        center_col, center_row = aoi['center']
        side_length = aoi['side_length']
        
        half_side = side_length // 2
        row_start = center_row - half_side
        row_end = center_row + half_side + (side_length % 2)
        col_start = center_col - half_side
        col_end = center_col + half_side + (side_length % 2)
        
        # Convert to map coordinates (using base_masked_transform)
        x_start, y_start = pixel_to_map(col_start, row_start, base_masked_transform)
        x_end, y_end = pixel_to_map(col_end, row_end, base_masked_transform)
        
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
    
    # Draw automatic registration extent outline
    if automatic_extent_corners is not None:
        corners = automatic_extent_corners
        # Close the polygon by adding first point at end
        corners_closed = corners + [corners[0]]
        x_coords = [c[0] for c in corners_closed]
        y_coords = [c[1] for c in corners_closed]
        ax.plot(x_coords, y_coords, color='yellow', linewidth=3, 
                linestyle='--', label='The Rectified image extent', alpha=0.8)
    
    # Set axis limits to match masked image extent
    ax.set_xlim(mask_left, mask_right)
    ax.set_ylim(mask_bottom, mask_top)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_title('Base Masked Image with AOI Outlines', fontsize=14, fontweight='bold', pad=10, y=1.02)
    
    # Create custom legend with darker yellow for rectified image extent
    handles, labels = ax.get_legend_handles_labels()
    # Find the rectified image extent entry and make it darker yellow
    for i, label in enumerate(labels):
        if label == 'The Rectified image extent':
            # Create a darker yellow color (mix yellow with a bit of orange/brown)
            darker_yellow = '#CCCC00'  # Darker yellow/olive color
            handles[i] = plt.Line2D([0], [0], color=darker_yellow, linewidth=3, 
                                   linestyle='--', alpha=0.8)
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=4, frameon=False, fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'aoi_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()
    print()
    
    # Step 14: Create 4×3 comparison grid
    print("Step 14: Creating 4×3 comparison grid...")
    
    # Prepare all images for comparison with RGB bands
    # Base masked: RGB 543 (bands 5, 4, 3)
    # Others: RGB 432 (bands 4, 3, 2)
    comparison_data = {}
    
    # Base masked (reference) - RGB 543: bands 5, 4, 3
    if base_masked_rgb is not None and base_masked_count >= 3:
        # Use masked image RGB data directly
        comparison_data['base_masked'] = {
            'bands': {
                'R': base_masked_rgb[0],  # Band 5 (NIR) -> Red
                'G': base_masked_rgb[1],   # Band 4 (Red) -> Green
                'B': base_masked_rgb[2]    # Band 3 (Green) -> Blue
            },
            'transform': base_masked_transform,
            'crs': base_masked_crs
        }
    elif base_raw_count >= 3:
        # Extract corresponding region from raw image and apply mask
        base_b5 = base_raw_data[0][raw_row_start:raw_row_end, raw_col_start:raw_col_end]  # Band 5
        base_b4 = base_raw_data[1][raw_row_start:raw_row_end, raw_col_start:raw_col_end]  # Band 4
        base_b3 = base_raw_data[2][raw_row_start:raw_row_end, raw_col_start:raw_col_end]  # Band 3
        
        if base_b5.shape == base_masked_mask.shape:
            mask_region = base_masked_mask > 0
            base_b5 = base_b5 * mask_region
            base_b4 = base_b4 * mask_region
            base_b3 = base_b3 * mask_region
        
        # Create transform for the extracted region
        region_transform = Affine(
            base_raw_transform[0], base_raw_transform[1],
            base_raw_transform[2] + raw_col_start * base_raw_transform[0],
            base_raw_transform[3], base_raw_transform[4],
            base_raw_transform[5] + raw_row_start * base_raw_transform[4]
        )
        comparison_data['base_masked'] = {
            'bands': {
                'R': base_b5,  # Band 5 -> Red
                'G': base_b4,  # Band 4 -> Green
                'B': base_b3   # Band 3 -> Blue
            },
            'transform': region_transform,
            'crs': base_raw_crs
        }
    
    # Homework 5 (map-to-image) - RGB 432: bands 4, 3, 2
    if 'B4' in homework_5_bands and 'B3' in homework_5_bands and 'B2' in homework_5_bands:
        comparison_data['homework_5'] = {
            'bands': {
                'R': homework_5_bands['B4']['data'],  # Band 4 (NIR) -> Red
                'G': homework_5_bands['B3']['data'],  # Band 3 (Red) -> Green
                'B': homework_5_bands['B2']['data']   # Band 2 (Green) -> Blue
            },
            'transform': homework_5_bands['B4']['transform'],
            'crs': homework_5_bands['B4']['crs']
        }
    
    # Manual registration - RGB 432: bands 4, 3, 2
    comparison_data['manual'] = {
        'bands': {
            'R': manual_registered_bands['B4']['data'],  # Band 4 (NIR) -> Red
            'G': manual_registered_bands['B3']['data'],  # Band 3 (Red) -> Green
            'B': manual_registered_bands['B2']['data']   # Band 2 (Green) -> Blue
        },
        'transform': manual_transform,
        'crs': base_raw_crs
    }
    
    # Automatic registration - RGB 432: bands 4, 3, 2
    comparison_data['automatic'] = {
        'bands': {
            'R': automatic_registered_bands['B4']['data'],  # Band 4 (NIR) -> Red
            'G': automatic_registered_bands['B3']['data'],  # Band 3 (Red) -> Green
            'B': automatic_registered_bands['B2']['data']   # Band 2 (Green) -> Blue
        },
        'transform': automatic_transform,
        'crs': base_raw_crs
    }
    
    # Create figure
    fig, axes = plt.subplots(4, 3, figsize=(11, 14))
    
    method_names = ['Base Masked', 'Homework 5 (Map-to-Image)', 'Manual (Image-to-Image)', 'Automatic (Image-to-Image)']
    method_keys = ['base_masked', 'homework_5', 'manual', 'automatic']
    
    # Convert AOI pixel coordinates to map coordinates first
    # Then convert to pixel coordinates for each method
    aoi_rgb_regions_by_method = {}
    
    # First, convert AOI definitions from base masked pixel coords to map coords
    aoi_map_coords = []
    for aoi in aoi_config:
        center_col, center_row = aoi['center']
        side_length_pixels = aoi['side_length']
        
        # Convert center to map coordinates
        center_map_x, center_map_y = pixel_to_map(center_col, center_row, base_masked_transform)
        
        # Convert side length from pixels to map coordinates (meters)
        # Calculate pixel size from transform
        pixel_size_x = abs(base_masked_transform[0])
        pixel_size_y = abs(base_masked_transform[4])
        side_length_meters = side_length_pixels * pixel_size_x  # Use average or x size
        
        aoi_map_coords.append({
            'center_map_x': center_map_x,
            'center_map_y': center_map_y,
            'side_length_meters': side_length_meters,
            'name': aoi['name']
        })
    
    # Extract RGB regions from each method using map coordinates
    for method_key in method_keys:
        if method_key not in comparison_data:
            continue
        
        bands_dict = comparison_data[method_key]['bands']
        transform = comparison_data[method_key]['transform']
        aoi_rgb_regions_by_method[method_key] = []
        
        # Get pixel size for this method
        method_pixel_size_x = abs(transform[0])
        method_pixel_size_y = abs(transform[4])
        
        # Get reference band shape (all bands should have same shape)
        ref_band = list(bands_dict.values())[0]
        
        for aoi_idx, aoi_map in enumerate(aoi_map_coords):
            # Convert map center to this method's pixel coordinates
            method_col, method_row = map_to_pixel(aoi_map['center_map_x'], aoi_map['center_map_y'], transform)
            
            # Convert side length from meters to pixels for this method
            side_length_pixels = int(aoi_map['side_length_meters'] / method_pixel_size_x)
            
            # Extract RGB regions
            rgb_regions = {}
            if 0 <= method_row < ref_band.shape[0] and 0 <= method_col < ref_band.shape[1] and side_length_pixels > 0:
                for color, band_data in bands_dict.items():
                    region, _ = extract_square_region(band_data, int(method_row), int(method_col), side_length_pixels)
                    rgb_regions[color] = region
                aoi_rgb_regions_by_method[method_key].append(rgb_regions)
            else:
                # Create empty region if out of bounds
                fallback_size = aoi_config[aoi_idx]['side_length']
                empty_region = np.zeros((fallback_size, fallback_size))
                aoi_rgb_regions_by_method[method_key].append({
                    'R': empty_region,
                    'G': empty_region,
                    'B': empty_region
                })
                print(f"  [WARNING] AOI {aoi_idx + 1} out of bounds for {method_key}: row={method_row:.1f}, col={method_col:.1f}, data_shape={ref_band.shape}")
    
    # Apply color balance correction to match row 1 (base_masked) appearance
    print("  Applying color balance correction to match reference (row 1)...")
    if 'base_masked' in aoi_rgb_regions_by_method:
        # Compute reference statistics from row 1 (base_masked)
        ref_stats = compute_channel_stats(aoi_rgb_regions_by_method['base_masked'], 
                                          percentile_low=2, percentile_high=98)
        
        # Apply correction to rows 2-4 (homework_5, manual, automatic)
        target_methods = ['homework_5', 'manual', 'automatic']
        for method_key in target_methods:
            if method_key in aoi_rgb_regions_by_method:
                # Compute target statistics
                target_stats = compute_channel_stats(aoi_rgb_regions_by_method[method_key],
                                                   percentile_low=2, percentile_high=98)
                
                # Apply color balance correction to each AOI
                for aoi_idx in range(len(aoi_rgb_regions_by_method[method_key])):
                    rgb_regions = aoi_rgb_regions_by_method[method_key][aoi_idx]
                    for color in ['R', 'G', 'B']:
                        corrected = apply_color_balance(
                            rgb_regions[color],
                            target_stats[color],
                            ref_stats[color]
                        )
                        aoi_rgb_regions_by_method[method_key][aoi_idx][color] = corrected
        
        print("  [OK] Color balance correction applied")
    
    # Calculate percentile-based min/max for each band and each AOI (column) across all methods (rows)
    # This ensures each column and each color channel has consistent scaling, avoiding outliers
    aoi_band_min_max = []  # List of dicts: [{'R': (min, max), 'G': (min, max), 'B': (min, max)}, ...]
    num_aois = len(aoi_config)
    percentile_low = 2   # Lower percentile to avoid dark outliers
    percentile_high = 98  # Upper percentile to avoid bright outliers
    
    for aoi_idx in range(num_aois):
        # Collect all valid pixel values for each band across all methods
        band_all_values = {'R': [], 'G': [], 'B': []}
        
        for method_key in method_keys:
            if method_key in aoi_rgb_regions_by_method:
                if aoi_idx < len(aoi_rgb_regions_by_method[method_key]):
                    rgb_regions = aoi_rgb_regions_by_method[method_key][aoi_idx]
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
    
    # Process each method (rows)
    for row_idx, (method_name, method_key) in enumerate(zip(method_names, method_keys)):
        if method_key not in aoi_rgb_regions_by_method:
            # Skip if method not available
            for col_idx in range(3):
                ax = axes[row_idx, col_idx]
                ax.axis('off')
            continue
        
        rgb_regions_list = aoi_rgb_regions_by_method[method_key]
        
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
            
            # Add colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(aoi_colors[col_idx])
                spine.set_linewidth(3)
            
            # Add labels
            if row_idx == 0:
                ax.set_title(f"{aoi_config[col_idx]['name']}", fontsize=12, fontweight='bold', pad=10)
            
            if col_idx == 0:
                ax.set_ylabel(method_name, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'method_comparison_4x3.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()
    print()
    
    # Final summary
    print("=" * 80)
    print("IMAGE-TO-IMAGE REGISTRATION SUMMARY")
    print("=" * 80)
    print(f"\n1. Manual Registration:")
    print(f"   - GCPs: {len(manual_gcps)} points")
    print(f"   - RMSE: {manual_rmse_total}")
    print(f"   - Output: registered_manual_B543.tif")
    
    print(f"\n2. Automatic Registration:")
    print(f"   - GCPs: {len(automatic_gcps)} points")
    print(f"   - RMSE: {automatic_rmse_total}")
    print(f"   - Output: registered_automatic_B543.tif")
    
    print(f"\n3. Visualizations:")
    print(f"   - Base raw with mask outline: base_with_mask_outline.png")
    print(f"   - Base masked only: base_masked_only.png")
    print(f"   - Manual registration with roads: manual_registration_with_roads.png")
    print(f"   - Automatic registration with roads: automatic_registration_with_roads.png")
    print(f"   - AOI overview: aoi_overview.png")
    print(f"   - Method comparison grid: method_comparison_4x3.png")
    
    print("\n" + "=" * 80)
    print("[OK] All processing completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

