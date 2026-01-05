"""
Calculate Reflectance Difference Between Two Time Periods

This script:
1. Reads the two mean reflectance GeoTIFF files (2021 and 2015)
2. Calculates pixel-wise difference (newer year - older year)
3. Saves the difference as a single-band GeoTIFF for change detection
"""

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from pathlib import Path
import json

# Get script directory and paths
script_dir = Path(__file__).parent.absolute()
homework_dir = script_dir.parent
data_dir = homework_dir / 'data'
input_dir = data_dir / 'change_detection_images'
output_dir = data_dir / 'change_detection_images'
selected_pair_file = data_dir / 'selected_pair.json'

print("=" * 60)
print("CALCULATE REFLECTANCE DIFFERENCE (CHANGE DETECTION)")
print("=" * 60)
print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print("=" * 60)
print()

# Load selected pair to identify which image is which year
if selected_pair_file.exists():
    with open(selected_pair_file, 'r') as f:
        pair_data = json.load(f)
    
    image_1 = pair_data['selected_pair']['image_1']
    image_2 = pair_data['selected_pair']['image_2']
    
    # Identify newer and older images
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
    
    print(f"Newer image: {newer_scene_id} ({newer_year})")
    print(f"Older image: {older_scene_id} ({older_year})")
    print()
else:
    print("Warning: Could not find selected_pair.json, will infer from filenames")
    newer_scene_id = "LC81660342021231LGN00"  # 2021
    older_scene_id = "LC81660342015231LGN01"  # 2015
    newer_year = 2021
    older_year = 2015

# Find mean reflectance files
newer_file = input_dir / f"{newer_scene_id}_mean_reflectance.tif"
older_file = input_dir / f"{older_scene_id}_mean_reflectance.tif"

if not newer_file.exists():
    print(f"Error: Newer year mean reflectance file not found: {newer_file}")
    exit(1)

if not older_file.exists():
    print(f"Error: Older year mean reflectance file not found: {older_file}")
    exit(1)

print(f"Reading newer image: {newer_file.name}")
print(f"Reading older image: {older_file.name}")
print()

# Read both images
with rasterio.open(newer_file) as src_newer:
    newer_data = src_newer.read(1)
    newer_profile = src_newer.profile.copy()
    newer_crs = src_newer.crs
    newer_transform = src_newer.transform
    newer_shape = (src_newer.height, src_newer.width)
    
    print(f"Newer image ({newer_year}):")
    print(f"  Size: {src_newer.width} × {src_newer.height} pixels")
    print(f"  CRS: {src_newer.crs}")
    print(f"  Data type: {src_newer.dtypes[0]}")
    print()

with rasterio.open(older_file) as src_older:
    older_data = src_older.read(1)
    older_profile = src_older.profile.copy()
    older_crs = src_older.crs
    older_transform = src_older.transform
    older_shape = (src_older.height, src_older.width)
    
    print(f"Older image ({older_year}):")
    print(f"  Size: {src_older.width} × {src_older.height} pixels")
    print(f"  CRS: {src_older.crs}")
    print(f"  Data type: {src_older.dtypes[0]}")
    print()

# Check spatial alignment
crs_match = newer_crs == older_crs
transform_match = newer_transform == older_transform
shape_match = newer_shape == older_shape

print("=" * 60)
print("SPATIAL ALIGNMENT CHECK")
print("=" * 60)
print(f"CRS match: {crs_match}")
print(f"Transform match: {transform_match}")
print(f"Shape match: {shape_match}")
print()

# Handle alignment
if crs_match and transform_match and shape_match:
    # Perfect alignment - direct subtraction
    print("Images are perfectly aligned. Performing direct subtraction...")
    difference = newer_data.astype(np.float32) - older_data.astype(np.float32)
    output_profile = newer_profile.copy()
    output_transform = newer_transform
    
elif crs_match and shape_match:
    # Same CRS and shape, but transform might differ slightly
    # Use newer image as reference
    print("Images have same CRS and shape. Using newer image as reference...")
    difference = newer_data.astype(np.float32) - older_data.astype(np.float32)
    output_profile = newer_profile.copy()
    output_transform = newer_transform
    
else:
    # Need to reproject/resample
    print("Images need spatial alignment. Reprojecting older image to newer image grid...")
    print("  This may take a moment...")
    
    # Reproject older image to match newer image
    older_reprojected = np.empty(newer_shape, dtype=np.float32)
    
    reproject(
        source=older_data.astype(np.float32),
        destination=older_reprojected,
        src_transform=older_transform,
        src_crs=older_crs,
        dst_transform=newer_transform,
        dst_crs=newer_crs,
        resampling=Resampling.bilinear
    )
    
    difference = newer_data.astype(np.float32) - older_reprojected
    output_profile = newer_profile.copy()
    output_transform = newer_transform

# Update profile for output
output_profile.update(
    count=1,
    dtype=rasterio.float32,
    compress='lzw',
    nodata=None
)

# Create output filename
output_file = output_dir / f"reflectance_difference_{newer_year}-{older_year}.tif"

print()
print("=" * 60)
print("CALCULATING DIFFERENCE")
print("=" * 60)
print(f"Formula: Difference = {newer_year} - {older_year}")
print(f"  Positive values = increase in reflectance")
print(f"  Negative values = decrease in reflectance")
print()

# Write difference to output file
print(f"Writing difference to: {output_file.name}")
with rasterio.open(output_file, 'w', **output_profile) as dst:
    dst.write(difference, 1)
    dst.set_band_description(1, f'Reflectance Difference ({newer_year} - {older_year})')

# Calculate and display statistics
valid_data = difference[~np.isnan(difference)]

if len(valid_data) > 0:
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    mean_val = np.mean(valid_data)
    std_val = np.std(valid_data)
    
    # Count positive and negative changes
    positive_count = np.sum(valid_data > 0)
    negative_count = np.sum(valid_data < 0)
    no_change_count = np.sum(valid_data == 0)
    total_pixels = len(valid_data)
    
    print()
    print("=" * 60)
    print("DIFFERENCE STATISTICS")
    print("=" * 60)
    print(f"Min difference: {min_val:.6f}")
    print(f"Max difference: {max_val:.6f}")
    print(f"Mean difference: {mean_val:.6f}")
    print(f"Std Dev: {std_val:.6f}")
    print()
    print(f"Change detection summary:")
    print(f"  Pixels with increase: {positive_count:,} ({100*positive_count/total_pixels:.2f}%)")
    print(f"  Pixels with decrease: {negative_count:,} ({100*negative_count/total_pixels:.2f}%)")
    print(f"  Pixels with no change: {no_change_count:,} ({100*no_change_count/total_pixels:.2f}%)")
    print(f"  Total valid pixels: {total_pixels:,}")
else:
    print("  [WARNING] No valid data in difference calculation")

print()
print("=" * 60)
print("DIFFERENCE CALCULATION COMPLETE")
print("=" * 60)
print(f"Output file: {output_file.name}")
print(f"Output location: {output_dir}")
print()
print("Difference file contains:")
print(f"  - Pixel-wise difference ({newer_year} - {older_year})")
print("  - Positive values indicate increase in reflectance")
print("  - Negative values indicate decrease in reflectance")
print("  - Float32 data type")
print("  - Same georeferencing as newer image")
print("=" * 60)

