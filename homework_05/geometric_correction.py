"""
Geometric Correction and Resampling Method Comparison

This script performs geometric correction of satellite images using ground control 
points (GCPs) and evaluates the effects of different resampling methods on image quality.

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
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies for road overlay
try:
    import fiona
    from shapely.geometry import shape
    from shapely.ops import transform as shapely_transform
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
base_dir = Path(__file__).parent
unrectified_dir = base_dir / 'Shiraz_Georeferencing' / 'Unrectified'
road_dir = base_dir / 'Shiraz_Georeferencing' / 'Road'
points_file = base_dir / 'points.pts'
output_dir = base_dir / 'output'

# Create output directory
output_dir.mkdir(exist_ok=True, parents=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_gcp_file(points_file):
    """
    Parse ENVI format GCP file.
    Format: Map (x,y), Image (x,y)
    Returns: List of GCPs and target CRS
    """
    gcps = []
    target_crs = None
    
    with open(points_file, 'r') as f:
        lines = f.readlines()
    
    # Parse header for CRS information
    for line in lines:
        if 'projection info' in line.lower():
            # Extract UTM zone and datum
            if 'UTM' in line and '39' in line and 'North' in line:
                target_crs = CRS.from_epsg(32639)  # UTM Zone 39N
                break
    
    # Parse GCP coordinates
    for line in lines:
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith(';'):
            continue
        
        # Parse coordinates: Map X, Map Y, Image X, Image Y
        parts = line.split()
        if len(parts) >= 4:
            try:
                map_x = float(parts[0])
                map_y = float(parts[1])
                img_x = float(parts[2])
                img_y = float(parts[3])
                
                gcps.append({
                    'map_x': map_x,
                    'map_y': map_y,
                    'img_x': img_x,
                    'img_y': img_y
                })
            except ValueError:
                continue
    
    return gcps, target_crs


def calculate_rmse(gcps_list, transform):
    """
    Calculate RMSE for GCPs.
    Transform image coordinates to map coordinates and compare with actual map coordinates.
    """
    errors_x = []
    errors_y = []
    errors_total = []
    
    # Get the transform (affine transformation matrix)
    # For forward transform: map_x = a * img_x + b * img_y + c
    #                        map_y = d * img_x + e * img_y + f
    a, b, c, d, e, f = transform[0], transform[1], transform[2], transform[3], transform[4], transform[5]
    
    for gcp in gcps_list:
        img_x, img_y = gcp['img_x'], gcp['img_y']
        map_x_true, map_y_true = gcp['map_x'], gcp['map_y']
        
        # Transform image coordinates to map coordinates
        map_x_pred = a * img_x + b * img_y + c
        map_y_pred = d * img_x + e * img_y + f
        
        # Calculate errors
        error_x = map_x_pred - map_x_true
        error_y = map_y_pred - map_y_true
        error_total = np.sqrt(error_x**2 + error_y**2)
        
        errors_x.append(error_x)
        errors_y.append(error_y)
        errors_total.append(error_total)
    
    # Calculate RMSE
    rmse_x = np.sqrt(np.mean(np.array(errors_x)**2))
    rmse_y = np.sqrt(np.mean(np.array(errors_y)**2))
    rmse_total = np.sqrt(np.mean(np.array(errors_total)**2))
    
    return rmse_x, rmse_y, rmse_total, errors_x, errors_y, errors_total


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
    row_end = center_row + half_side + (side_length % 2)  # Handle odd side_length
    col_start = center_col - half_side
    col_end = center_col + half_side + (side_length % 2)
    
    # Clamp to image bounds
    row_start = max(0, row_start)
    row_end = min(height, row_end)
    col_start = max(0, col_start)
    col_end = min(width, col_end)
    
    # Extract region
    region = band_data[row_start:row_end, col_start:col_end]
    
    return region, (row_start, row_end, col_start, col_end)


def map_to_pixel(x, y, transform):
    """Convert map coordinates to pixel coordinates."""
    col = (x - transform[2]) / transform[0]
    row = (y - transform[5]) / transform[4]
    return col, row


def pixel_to_map(col, row, transform):
    """Convert pixel coordinates to map coordinates."""
    x = transform[0] * col + transform[2]
    y = transform[4] * row + transform[5]
    return x, y


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main function to execute geometric correction workflow."""
    
    print("=" * 80)
    print("GEOMETRIC CORRECTION AND RESAMPLING METHOD COMPARISON")
    print("=" * 80)
    print()
    
    # Step 1: Load unrectified images
    print("Step 1: Loading unrectified images...")
    bands = {}
    band_files = {
        'B2': unrectified_dir / 'B2.TIF',
        'B3': unrectified_dir / 'B3.TIF',
        'B4': unrectified_dir / 'B4.TIF'
    }
    
    for band_name, file_path in band_files.items():
        if file_path.exists():
            with rasterio.open(file_path) as src:
                bands[band_name] = {
                    'data': src.read(1),
                    'profile': src.profile.copy(),
                    'width': src.width,
                    'height': src.height
                }
            print(f"  [OK] Loaded {band_name}: {bands[band_name]['width']} x {bands[band_name]['height']} pixels")
        else:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Use B4 dimensions as reference
    ref_height, ref_width = bands['B4']['height'], bands['B4']['width']
    print(f"  Reference image dimensions: {ref_width} x {ref_height} pixels\n")
    
    # Step 2: Parse GCPs
    print("Step 2: Parsing Ground Control Points...")
    gcps_list, dst_crs = parse_gcp_file(points_file)
    
    if dst_crs is None:
        dst_crs = CRS.from_epsg(32639)  # Default to UTM Zone 39N
    
    print(f"  [OK] Parsed {len(gcps_list)} GCPs")
    print(f"  [OK] Target CRS: {dst_crs}\n")
    
    # Create GCP dataframe for reporting
    gcp_df = pd.DataFrame(gcps_list)
    gcp_df.index = range(1, len(gcp_df) + 1)
    
    # Step 3: Create rasterio GCP objects
    print("Step 3: Creating rasterio GCP objects...")
    rasterio_gcps = []
    for gcp in gcps_list:
        gcp_obj = GroundControlPoint(
            row=gcp['img_y'],  # row in image
            col=gcp['img_x'],  # column in image
            x=gcp['map_x'],    # map X (easting)
            y=gcp['map_y'],    # map Y (northing)
            z=0.0              # elevation (not used)
        )
        rasterio_gcps.append(gcp_obj)
    
    print(f"  [OK] Created {len(rasterio_gcps)} rasterio GCP objects\n")
    
    # Step 4: Calculate transform
    print("Step 4: Calculating transformation...")
    transform, width, height = calculate_default_transform(
        src_crs=None,  # No source CRS
        dst_crs=dst_crs,
        width=ref_width,
        height=ref_height,
        gcps=rasterio_gcps,
        gcps_crs=dst_crs
    )
    
    pixel_size_x = abs(transform[0])
    pixel_size_y = abs(transform[4])
    print(f"  [OK] Transform calculated")
    print(f"  Output dimensions: {width} x {height} pixels")
    print(f"  Pixel size: {pixel_size_x:.2f} x {pixel_size_y:.2f} meters\n")
    
    # Step 5: Perform geometric correction for all bands
    print("Step 5: Performing geometric correction for all bands...")
    corrected_bands = {}
    
    for band_name in ['B2', 'B3', 'B4']:
        print(f"  Processing {band_name}...")
        src_array = bands[band_name]['data']
        
        # Create destination array
        dst_array = np.zeros((height, width), dtype=src_array.dtype)
        
        # Create a temporary in-memory dataset with GCPs
        # Create source profile with GCPs
        src_profile = {
            'driver': 'GTiff',
            'width': ref_width,
            'height': ref_height,
            'count': 1,
            'dtype': src_array.dtype
        }
        
        # Use MemoryFile with explicit GCP setting after opening
        with MemoryFile() as memfile:
            with memfile.open(**src_profile) as src_ds:
                src_ds.write(src_array, 1)
                # Set GCPs after writing
                src_ds.gcps = (rasterio_gcps, dst_crs)
                
                # Perform reprojection using the dataset with GCPs
                reproject(
                    source=rasterio.band(src_ds, 1),
                    destination=dst_array,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest  # Use nearest for initial correction
                )
        
        # Create destination profile
        dst_profile = {
            'driver': 'GTiff',
            'width': width,
            'height': height,
            'count': 1,
            'dtype': src_array.dtype,
            'crs': dst_crs,
            'transform': transform,
            'compress': 'lzw'
        }
        
        corrected_bands[band_name] = {
            'data': dst_array,
            'profile': dst_profile,
            'transform': transform,
            'crs': dst_crs
        }
        
        # Save corrected band
        output_path = output_dir / f'corrected_{band_name}.tif'
        with rasterio.open(output_path, 'w', **dst_profile) as dst:
            dst.write(dst_array, 1)
        print(f"    [OK] Saved {output_path}")
    
    print("  [OK] Geometric correction completed for all bands\n")
    
    # Step 6: Calculate RMSE
    print("Step 6: Calculating RMSE...")
    rmse_x, rmse_y, rmse_total, errors_x, errors_y, errors_total = calculate_rmse(gcps_list, transform)
    
    print(f"  RMSE X (Easting): {rmse_x:.4f} meters")
    print(f"  RMSE Y (Northing): {rmse_y:.4f} meters")
    print(f"  Total RMSE: {rmse_total:.4f} meters")
    print(f"  Total RMSE: {rmse_total/pixel_size_x:.4f} pixels (assuming ~{pixel_size_x:.1f}m pixel size)\n")
    
    # Add errors to GCP dataframe
    gcp_df['Error_X (m)'] = errors_x
    gcp_df['Error_Y (m)'] = errors_y
    gcp_df['Total_Error (m)'] = errors_total
    
    # Save RMSE report
    report_path = output_dir / 'gcp_rmse_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GCP RMSE REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of GCPs: {len(gcps_list)}\n")
        f.write(f"Target CRS: {dst_crs}\n\n")
        f.write(f"RMSE X (Easting): {rmse_x:.4f} meters\n")
        f.write(f"RMSE Y (Northing): {rmse_y:.4f} meters\n")
        f.write(f"Total RMSE: {rmse_total:.4f} meters\n")
        f.write(f"Total RMSE: {rmse_total/pixel_size_x:.4f} pixels\n\n")
        f.write("GCP Table with Individual Errors:\n")
        f.write("-" * 80 + "\n")
        f.write(gcp_df.to_string())
        f.write("\n")
    print(f"  [OK] Saved RMSE report: {report_path}\n")
    
    # Step 7: Apply resampling methods to Band 4
    print("Step 7: Applying resampling methods to Band 4...")
    resampling_methods = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic
    }
    
    resampled_bands = {}
    source_b4_data = bands['B4']['data']
    
    for method_name, resampling in resampling_methods.items():
        print(f"  Processing {method_name}...")
        
        # Create destination array
        dst_array = np.zeros((height, width), dtype=source_b4_data.dtype)
        
        # Create a temporary in-memory dataset with GCPs
        src_profile = {
            'driver': 'GTiff',
            'width': ref_width,
            'height': ref_height,
            'count': 1,
            'dtype': source_b4_data.dtype
        }
        
        with MemoryFile() as memfile:
            with memfile.open(**src_profile) as src_ds:
                src_ds.write(source_b4_data, 1)
                # Set GCPs after writing
                src_ds.gcps = (rasterio_gcps, dst_crs)
                
                # Perform geometric correction with specific resampling method
                reproject(
                    source=rasterio.band(src_ds, 1),
                    destination=dst_array,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling
                )
        
        resampled_bands[method_name] = {
            'data': dst_array,
            'profile': corrected_bands['B4']['profile'].copy(),
            'transform': transform,
            'crs': dst_crs
        }
        
        # Save individual resampled band
        output_path = output_dir / f'resampled_B4_{method_name}.tif'
        with rasterio.open(output_path, 'w', **resampled_bands[method_name]['profile']) as dst:
            dst.write(dst_array, 1)
        print(f"    [OK] Saved {output_path}")
    
    print("  [OK] Resampling completed for all methods\n")
    
    # Step 8: Create stacked GeoTIFF
    print("Step 8: Creating stacked GeoTIFF...")
    stacked_profile = resampled_bands['nearest']['profile'].copy()
    stacked_profile['count'] = 3  # Three bands: nearest, bilinear, cubic
    
    stacked_path = output_dir / 'resampled_B4_stacked.tif'
    with rasterio.open(stacked_path, 'w', **stacked_profile) as dst:
        dst.write(resampled_bands['nearest']['data'], 1)
        dst.write(resampled_bands['bilinear']['data'], 2)
        dst.write(resampled_bands['cubic']['data'], 3)
    
    print(f"  [OK] Saved stacked GeoTIFF: {stacked_path}")
    print("    Band 1: Nearest Neighbor")
    print("    Band 2: Bilinear")
    print("    Band 3: Cubic Convolution\n")
    
    # Step 9: Load road shapefile
    print("Step 9: Loading road shapefile...")
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
                
                image_crs = corrected_bands['B4']['crs']
                if road_crs != image_crs:
                    print(f"  Note: Road CRS differs from image CRS, will reproject")
                else:
                    print("  [OK] CRS matches image CRS")
            except Exception as e:
                print(f"  [ERROR] Error loading shapefile: {e}")
                roads = None
        else:
            print(f"  [ERROR] Shapefile not found: {road_shp}")
    else:
        print("  [SKIP] Road overlay skipped (fiona/shapely/pyproj not available)\n")
    
    # Step 10: Create RGB composite and overlay roads
    print("Step 10: Creating RGB composite with road overlay...")
    rgb_composite = create_rgb_composite(corrected_bands)
    
    if rgb_composite is not None:
        img_transform = corrected_bands['B4']['transform']
        img_crs = corrected_bands['B4']['crs']
        
        # Calculate extent based on image bounds only
        left = img_transform[2]
        top = img_transform[5]
        right = left + width * img_transform[0]
        bottom = top + height * img_transform[4]
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Display RGB composite with image extent
        ax.imshow(rgb_composite, extent=[left, right, bottom, top])
        
        # Set axis limits to match image extent exactly (crop to image bounds)
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        
        # Overlay roads if available
        if roads is not None:
            try:
                # Reproject roads if needed
                if road_crs != img_crs:
                    project = pyproj.Transformer.from_crs(
                        pyproj.CRS(str(road_crs)),
                        pyproj.CRS(str(img_crs)),
                        always_xy=True
                    ).transform
                else:
                    project = lambda x, y: (x, y)
                
                # Plot roads with high contrast (yellow/cyan color, thick, fully opaque)
                road_color = 'yellow'  # High contrast color for visibility
                for idx, road in enumerate(roads):
                    if road.geom_type == 'LineString':
                        coords = list(road.coords)
                        x_coords = [project(x, y)[0] for x, y in coords]
                        y_coords = [project(x, y)[1] for x, y in coords]
                        ax.plot(x_coords, y_coords, color=road_color, linewidth=3.0, alpha=1.0, 
                               label='Roads' if idx == 0 else '')
                    elif road.geom_type == 'MultiLineString':
                        for line in road.geoms:
                            coords = list(line.coords)
                            x_coords = [project(x, y)[0] for x, y in coords]
                            y_coords = [project(x, y)[1] for x, y in coords]
                            ax.plot(x_coords, y_coords, color=road_color, linewidth=3.0, alpha=1.0,
                                   label='Roads' if idx == 0 else '')
                
                ax.legend(loc='upper right')
            except Exception as e:
                print(f"  Note: Could not overlay roads: {e}")
        
        # Plot Ground Control Points (GCPs)
        print("  Plotting Ground Control Points...")
        gcp_x_coords = [gcp['map_x'] for gcp in gcps_list]
        gcp_y_coords = [gcp['map_y'] for gcp in gcps_list]
        ax.scatter(gcp_x_coords, gcp_y_coords, c='cyan', marker='o', s=100, 
                  edgecolors='darkblue', linewidths=2, alpha=1.0, 
                  label='GCPs', zorder=10)
        
        # Add legend entry for GCPs if roads weren't plotted
        if roads is None:
            ax.legend(loc='upper right')
        else:
            # Update legend to include GCPs
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper right')
        
        print(f"  [OK] Plotted {len(gcps_list)} GCPs")
        
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        ax.set_title('Rectified Image with Road Overlay\n(False Color: NIR-Red-Green)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'rectified_with_roads.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved visualization: {output_path}")
        plt.close()
    else:
        print("  [ERROR] Could not create RGB composite\n")
    
    # Step 11: Define AOIs
    print("Step 11: Defining Areas of Interest (AOIs)...")
    
    # Define AOIs in pixel coordinates (matching miscellaneous/visualize_rectification_comparison.py)
    # Format: {'center': (row, col), 'side_length': pixels}
    aoi_config = [
        {'center': (100, 150), 'side_length': 50},
        {'center': (210, 298), 'side_length': 50},  # Moved 2 pixels to the left
        {'center': (250, 175), 'side_length': 50},
    ]
    
    # Convert to format used by the script
    aois_pixel = []
    aoi_names = ['AOI 1', 'AOI 2', 'AOI 3']
    
    for idx, aoi in enumerate(aoi_config):
        center_row, center_col = aoi['center']
        side_length = aoi['side_length']
        
        aois_pixel.append({
            'name': aoi_names[idx],
            'center_row': center_row,
            'center_col': center_col,
            'side_length': side_length
        })
        
        print(f"  {aoi_names[idx]}:")
        print(f"    Pixel center: (row={center_row}, col={center_col})")
        print(f"    Side length: {side_length} pixels")
    
    print()
    
    # Step 12: Extract AOI regions
    print("Step 12: Extracting AOI regions from resampled bands...")
    aoi_regions = {}
    
    for aoi_idx, aoi in enumerate(aois_pixel):
        aoi_name = f"AOI_{aoi_idx + 1}"
        aoi_regions[aoi_name] = {}
        
        center_row = int(aoi['center_row'])
        center_col = int(aoi['center_col'])
        side_length = aoi['side_length']
        
        for method_name in ['nearest', 'bilinear', 'cubic']:
            region, bounds = extract_square_region(
                resampled_bands[method_name]['data'],
                center_row, center_col, side_length
            )
            aoi_regions[aoi_name][method_name] = {
                'data': region,
                'bounds': bounds
            }
        
        print(f"  [OK] Extracted regions for {aoi['name']}")
    
    print()
    
    # Step 13: Create 3x3 comparison grid
    print("Step 13: Creating 3x3 comparison grid...")
    
    method_names_display = {
        'nearest': 'Nearest Neighbor',
        'bilinear': 'Bilinear',
        'cubic': 'Cubic Convolution'
    }
    
    # Calculate global min/max for consistent colormap
    all_regions = []
    for aoi_name in aoi_regions:
        for method_name in ['nearest', 'bilinear', 'cubic']:
            all_regions.append(aoi_regions[aoi_name][method_name]['data'])
    
    global_min = min(region.min() for region in all_regions)
    global_max = max(region.max() for region in all_regions)
    print(f"  Global colormap range: {global_min:.2f} to {global_max:.2f}")
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    fig.suptitle('Resampling Method Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    # Colors for AOI borders
    aoi_colors = ['red', 'lime', 'cyan']
    
    # Process each AOI (rows)
    for row_idx, (aoi_name, aoi_info) in enumerate(aoi_regions.items()):
        aoi_display_name = aois_pixel[row_idx]['name']
        
        # Display each resampling method (columns)
        for col_idx, method_name in enumerate(['nearest', 'bilinear', 'cubic']):
            ax = axes[row_idx, col_idx]
            
            region = aoi_regions[aoi_name][method_name]['data']
            
            # Display the region with shared colormap scale
            ax.imshow(region, cmap='gray', aspect='equal', interpolation='nearest',
                     vmin=global_min, vmax=global_max)
            
            # Remove ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colored border matching the overview outline color
            for spine in ax.spines.values():
                spine.set_edgecolor(aoi_colors[row_idx])
                spine.set_linewidth(3)
            
            # Add labels
            if row_idx == 0:
                ax.set_title(method_names_display[method_name], fontsize=12, fontweight='bold', pad=10)
            
            if col_idx == 0:
                ax.set_ylabel(aoi_display_name, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'resampling_comparison_3x3.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved comparison grid: {output_path}")
    plt.close()
    
    # Step 14: Create overview image with AOI outlines
    print("Step 14: Creating overview image with AOI outlines...")
    
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    
    # Display Band 4 (using nearest neighbor resampled version)
    overview_data = resampled_bands['nearest']['data']
    ax.imshow(overview_data, cmap='gray', interpolation='nearest',
             extent=[left, right, bottom, top])
    
    # Draw AOI outlines
    for aoi_idx, aoi in enumerate(aois_pixel):
        center_col = aoi['center_col']
        center_row = aoi['center_row']
        side_length = aoi['side_length']
        
        # Calculate bounding box in pixel coordinates
        half_side = side_length // 2
        row_start = int(center_row - half_side)
        row_end = int(center_row + half_side + (side_length % 2))
        col_start = int(center_col - half_side)
        col_end = int(center_col + half_side + (side_length % 2))
        
        # Convert to map coordinates
        x_start, y_start = pixel_to_map(col_start, row_start, transform)
        x_end, y_end = pixel_to_map(col_end, row_end, transform)
        
        # Calculate width and height in map coordinates
        rect_width = x_end - x_start
        rect_height = y_end - y_start
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x_start, y_start),
            rect_width,
            rect_height,
            linewidth=2.5,
            edgecolor=aoi_colors[aoi_idx],
            facecolor='none',
            label=aois_pixel[aoi_idx]['name']
        )
        ax.add_patch(rect)
    
    # Remove axis, ticks, and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Title
    ax.set_title('Full Rectified Image with AOI Outlines', fontsize=14, fontweight='bold', pad=10, y=1.02)
    
    # Legend
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, 
             frameon=False, fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'aoi_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved overview image: {output_path}")
    plt.close()
    
    # Final summary
    print("\n" + "=" * 80)
    print("GEOMETRIC CORRECTION AND RESAMPLING SUMMARY")
    print("=" * 80)
    print(f"\n1. Ground Control Points:")
    print(f"   - Number of GCPs: {len(gcps_list)}")
    print(f"   - Target CRS: {dst_crs}")
    
    print(f"\n2. Geometric Correction:")
    print(f"   - Input dimensions: {ref_width} x {ref_height} pixels")
    print(f"   - Output dimensions: {width} x {height} pixels")
    print(f"   - Pixel size: {pixel_size_x:.2f} x {pixel_size_y:.2f} meters")
    
    print(f"\n3. RMSE Assessment:")
    print(f"   - RMSE X (Easting): {rmse_x:.4f} meters")
    print(f"   - RMSE Y (Northing): {rmse_y:.4f} meters")
    print(f"   - Total RMSE: {rmse_total:.4f} meters ({rmse_total/pixel_size_x:.4f} pixels)")
    
    print(f"\n4. Resampling Methods:")
    print(f"   - Nearest Neighbor: [OK]")
    print(f"   - Bilinear: [OK]")
    print(f"   - Cubic Convolution: [OK]")
    
    print(f"\n5. Output Files:")
    print(f"   - Corrected bands: corrected_B2.tif, corrected_B3.tif, corrected_B4.tif")
    print(f"   - Resampled bands: resampled_B4_nearest.tif, resampled_B4_bilinear.tif, resampled_B4_cubic.tif")
    print(f"   - Stacked GeoTIFF: resampled_B4_stacked.tif")
    print(f"   - Visualizations: rectified_with_roads.png, resampling_comparison_3x3.png, aoi_overview.png")
    print(f"   - Report: gcp_rmse_report.txt")
    
    print("\n" + "=" * 80)
    print("[OK] All processing completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

