"""
Create vegetation mask from NDVI layer

This script:
1. Loads or calculates NDVI (if not already calculated)
2. Applies threshold (NDVI > 0.35) to create binary vegetation mask
3. Exports vegetation mask visualization
4. Reports statistics (vegetated vs non-vegetated pixel counts)
"""

import ee
import os
import json
import urllib.request
from pathlib import Path
import numpy as np

# Clear terminal screen
os.system('cls' if os.name == 'nt' else 'clear')

# Initialize Google Earth Engine
try:
    ee.Initialize(project='surface-hydrology')
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing GEE: {e}")
    print("You may need to run: ee.Authenticate()")
    exit(1)

# Load scene information
scene_info_file = os.path.join('..', 'data', 'selected_scene.json')
if not os.path.exists(scene_info_file):
    print(f"Error: {scene_info_file} not found.")
    exit(1)

with open(scene_info_file, 'r') as f:
    scene_info = json.load(f)

scene_id = scene_info['scene_id']
collection_id = scene_info['collection_id']

print("=" * 60)
print("VEGETATION MASK CREATION")
print("=" * 60)
print(f"Scene ID: {scene_id}")
print(f"Collection: {collection_id}")
print("=" * 60)
print()

# Load the image
image = (ee.ImageCollection(collection_id)
         .filter(ee.Filter.eq('LANDSAT_SCENE_ID', scene_id))
         .first())

try:
    image_id = image.get('system:id').getInfo()
    print(f"Successfully loaded image: {image_id}")
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

aoi = image.geometry()
native_crs = image.select(0).projection().crs().getInfo()
print(f"Image native CRS: {native_crs}")
print()

# ===================================================================
# CALCULATE NDVI (if not already done)
# ===================================================================
print("=" * 60)
print("CALCULATING NDVI")
print("=" * 60)

# Extract reflectance metadata
try:
    refl_mult_b3 = image.get('REFLECTANCE_MULT_BAND_3').getInfo()
    refl_add_b3 = image.get('REFLECTANCE_ADD_BAND_3').getInfo()
    refl_mult_b4 = image.get('REFLECTANCE_MULT_BAND_4').getInfo()
    refl_add_b4 = image.get('REFLECTANCE_ADD_BAND_4').getInfo()
except Exception as e:
    try:
        props = image.getInfo()['properties']
        refl_mult_b3 = props.get('REFLECTANCE_MULT_BAND_3')
        refl_add_b3 = props.get('REFLECTANCE_ADD_BAND_3')
        refl_mult_b4 = props.get('REFLECTANCE_MULT_BAND_4')
        refl_add_b4 = props.get('REFLECTANCE_ADD_BAND_4')
    except Exception as e2:
        print(f"Error: Could not extract reflectance metadata: {e2}")
        exit(1)

# Convert to reflectance and calculate NDVI
band3 = image.select('B3')
band4 = image.select('B4')

red_reflectance = band3.multiply(refl_mult_b3).add(refl_add_b3).rename('red')
nir_reflectance = band4.multiply(refl_mult_b4).add(refl_add_b4).rename('nir')

ndvi = nir_reflectance.subtract(red_reflectance).divide(
    nir_reflectance.add(red_reflectance)
).rename('ndvi').clamp(-1, 1)

print("NDVI calculated successfully.")
print()

# ===================================================================
# CREATE VEGETATION MASK
# ===================================================================
print("=" * 60)
print("CREATING VEGETATION MASK")
print("=" * 60)

# Threshold: NDVI > 0.35 indicates vegetated areas
# Values: 1 = vegetated, 0 = non-vegetated
ndvi_threshold = 0.35
print(f"Using NDVI threshold: {ndvi_threshold}")
print("  NDVI > {:.2f} = Vegetated (1)".format(ndvi_threshold))
print("  NDVI <= {:.2f} = Non-vegetated (0)".format(ndvi_threshold))
print()

vegetation_mask = ndvi.gt(ndvi_threshold).rename('vegetation_mask')

# Calculate statistics
mask_stats = vegetation_mask.reduceRegion(
    reducer=ee.Reducer.frequencyHistogram().combine(
        ee.Reducer.count(), '', True
    ),
    geometry=aoi,
    scale=30,
    maxPixels=1e9
).getInfo()

# Extract pixel counts
total_pixels = mask_stats.get('vegetation_mask_count', 0)
histogram = mask_stats.get('vegetation_mask_histogram', {})

vegetated_pixels = histogram.get('1', 0)
non_vegetated_pixels = histogram.get('0', 0)

vegetated_percent = (vegetated_pixels / total_pixels * 100) if total_pixels > 0 else 0
non_vegetated_percent = (non_vegetated_pixels / total_pixels * 100) if total_pixels > 0 else 0

print("Vegetation Mask Statistics:")
print(f"  Total pixels: {total_pixels:,}")
print(f"  Vegetated pixels: {vegetated_pixels:,} ({vegetated_percent:.2f}%)")
print(f"  Non-vegetated pixels: {non_vegetated_pixels:,} ({non_vegetated_percent:.2f}%)")
print()

# ===================================================================
# VISUALIZE AND EXPORT MASK
# ===================================================================
print("=" * 60)
print("EXPORTING VEGETATION MASK")
print("=" * 60)

output_dir = Path(os.path.join('..', 'output'))
output_dir.mkdir(exist_ok=True, parents=True)

# Visualize mask: white = vegetated, black = non-vegetated
mask_visualized = vegetation_mask.visualize(
    min=0,
    max=1,
    palette=['black', 'white']
)

def download_image(image, filename, aoi, native_crs, scale=30, max_scale=120):
    """Download image with automatic scale adjustment."""
    scales_to_try = [scale, scale * 2, scale * 4, max_scale]
    
    for current_scale in scales_to_try:
        try:
            print(f"Attempting download at {current_scale}m resolution...")
            download_url = image.getDownloadURL({
                'scale': current_scale,
                'crs': native_crs,
                'region': aoi,
                'format': 'PNG'
            })
            
            print(f"Downloading to: {filename}")
            urllib.request.urlretrieve(download_url, str(filename))
            
            file_size = os.path.getsize(filename) / (1024 * 1024)
            print(f"✓ Image saved successfully: {filename}")
            print(f"  Resolution: {current_scale}m")
            print(f"  File size: {file_size:.2f} MB")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "must be less than or equal to" in error_msg:
                if current_scale < max_scale:
                    print(f"  Image too large at {current_scale}m, trying higher scale...")
                    continue
                else:
                    print(f"  Error: Image still too large even at {max_scale}m resolution.")
                    return False
            else:
                print(f"  Error: {e}")
                return False
    
    return False

mask_filename = output_dir / f"VegetationMask_{scene_id}.png"
print("Downloading vegetation mask...")
mask_success = download_image(mask_visualized, mask_filename, aoi, native_crs, scale=30, max_scale=120)

if not mask_success:
    mask_filename = None

print()

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("=" * 60)
print("VEGETATION MASK CREATION COMPLETE")
print("=" * 60)
print("Mask Statistics:")
print(f"  Threshold: NDVI > {ndvi_threshold}")
print(f"  Vegetated: {vegetated_pixels:,} pixels ({vegetated_percent:.2f}%)")
print(f"  Non-vegetated: {non_vegetated_pixels:,} pixels ({non_vegetated_percent:.2f}%)")
print()
print("Downloaded Files:")
if mask_filename:
    print(f"  ✓ Vegetation mask: {mask_filename}")
print(f"All files saved in: {output_dir.absolute()}")
print()

