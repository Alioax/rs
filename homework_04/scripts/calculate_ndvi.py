"""
Calculate NDVI (Normalized Difference Vegetation Index) for a selected Landsat 5 TM scene

This script:
1. Loads the scene ID from selected_scene.json (from Homework 3)
2. Loads the corresponding Landsat 5 TM image
3. Converts Red (B3) and NIR (B4) bands to TOA reflectance
4. Calculates NDVI = (NIR - Red) / (NIR + Red)
5. Exports NDVI map
"""

import ee
import os
import json
import urllib.request
from pathlib import Path

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

# Load scene information from JSON file
scene_info_file = os.path.join('..', 'data', 'selected_scene.json')
if not os.path.exists(scene_info_file):
    print(f"Error: {scene_info_file} not found.")
    print("Please ensure the scene data file exists.")
    exit(1)

with open(scene_info_file, 'r') as f:
    scene_info = json.load(f)

scene_id = scene_info['scene_id']
collection_id = scene_info['collection_id']
date_acquired = scene_info['date_acquired']
cloud_cover = scene_info['cloud_cover']

print("=" * 60)
print("NDVI CALCULATION")
print("=" * 60)
print(f"Scene ID: {scene_id}")
print(f"Date Acquired: {date_acquired}")
print(f"Cloud Cover: {cloud_cover}%")
print(f"Collection: {collection_id}")
print("=" * 60)
print()

# Load the specific image from the collection
image = (ee.ImageCollection(collection_id)
         .filter(ee.Filter.eq('LANDSAT_SCENE_ID', scene_id))
         .first())

# Verify the image exists
try:
    image_id = image.get('system:id').getInfo()
    print(f"Successfully loaded image: {image_id}")
except Exception as e:
    print(f"Error loading image: {e}")
    print("Trying alternative method...")
    system_id = scene_info.get('system_id', '')
    if system_id and system_id != 'N/A':
        image = ee.Image(system_id)
        print(f"Loaded image using system ID: {system_id}")
    else:
        print("Could not load the image. Please check the scene ID.")
        exit(1)

print()

# Get image geometry and native projection
aoi = image.geometry()
native_crs = image.select(0).projection().crs().getInfo()
print(f"Image native CRS: {native_crs}")
print()

# ===================================================================
# CONVERT REFLECTIVE BANDS TO TOA REFLECTANCE
# ===================================================================
print("=" * 60)
print("CONVERTING BANDS TO TOA REFLECTANCE")
print("=" * 60)

# Extract reflectance metadata parameters
print("Extracting reflectance metadata parameters...")
try:
    # For Landsat 5 TM Collection 2, use REFLECTANCE_MULT and REFLECTANCE_ADD
    # These directly convert DN to TOA reflectance (0-1 range)
    refl_mult_b3 = image.get('REFLECTANCE_MULT_BAND_3').getInfo()
    refl_add_b3 = image.get('REFLECTANCE_ADD_BAND_3').getInfo()
    refl_mult_b4 = image.get('REFLECTANCE_MULT_BAND_4').getInfo()
    refl_add_b4 = image.get('REFLECTANCE_ADD_BAND_4').getInfo()
    
    print(f"REFLECTANCE_MULT_BAND_3: {refl_mult_b3}")
    print(f"REFLECTANCE_ADD_BAND_3: {refl_add_b3}")
    print(f"REFLECTANCE_MULT_BAND_4: {refl_mult_b4}")
    print(f"REFLECTANCE_ADD_BAND_4: {refl_add_b4}")
    
except Exception as e:
    print(f"Error extracting reflectance metadata: {e}")
    print("Trying alternative property access...")
    try:
        props = image.getInfo()['properties']
        refl_mult_b3 = props.get('REFLECTANCE_MULT_BAND_3')
        refl_add_b3 = props.get('REFLECTANCE_ADD_BAND_3')
        refl_mult_b4 = props.get('REFLECTANCE_MULT_BAND_4')
        refl_add_b4 = props.get('REFLECTANCE_ADD_BAND_4')
        
        if not all([refl_mult_b3 is not None, refl_add_b3 is not None, 
                   refl_mult_b4 is not None, refl_add_b4 is not None]):
            raise ValueError("Required reflectance metadata properties not found")
        
        print(f"REFLECTANCE_MULT_BAND_3: {refl_mult_b3}")
        print(f"REFLECTANCE_ADD_BAND_3: {refl_add_b3}")
        print(f"REFLECTANCE_MULT_BAND_4: {refl_mult_b4}")
        print(f"REFLECTANCE_ADD_BAND_4: {refl_add_b4}")
    except Exception as e2:
        print(f"Error: Could not extract required reflectance metadata: {e2}")
        print("NDVI calculation cannot proceed without reflectance conversion.")
        exit(1)

print()

# Select Red (B3) and NIR (B4) bands
print("Selecting Red (B3) and NIR (B4) bands...")
band3 = image.select('B3')  # Red band
band4 = image.select('B4')  # NIR band

# Convert DN to TOA Reflectance
# Formula: ρ = REFLECTANCE_MULT_BAND_X * DN + REFLECTANCE_ADD_BAND_X
print("Converting DN to TOA reflectance...")
red_reflectance = band3.multiply(refl_mult_b3).add(refl_add_b3).rename('red')
nir_reflectance = band4.multiply(refl_mult_b4).add(refl_add_b4).rename('nir')

# Compute reflectance statistics for verification
red_stats = red_reflectance.reduceRegion(
    reducer=ee.Reducer.percentile([2, 98]),
    geometry=aoi,
    scale=30,
    maxPixels=1e9
).getInfo()

nir_stats = nir_reflectance.reduceRegion(
    reducer=ee.Reducer.percentile([2, 98]),
    geometry=aoi,
    scale=30,
    maxPixels=1e9
).getInfo()

red_min = red_stats.get('red_p2', 0)
red_max = red_stats.get('red_p98', 1)
nir_min = nir_stats.get('nir_p2', 0)
nir_max = nir_stats.get('nir_p98', 1)

print(f"Red reflectance range: {red_min:.4f} - {red_max:.4f}")
print(f"NIR reflectance range: {nir_min:.4f} - {nir_max:.4f}")
print()

# ===================================================================
# CALCULATE NDVI
# ===================================================================
print("=" * 60)
print("CALCULATING NDVI")
print("=" * 60)

# NDVI = (NIR - Red) / (NIR + Red)
print("Calculating NDVI = (NIR - Red) / (NIR + Red)...")
ndvi = nir_reflectance.subtract(red_reflectance).divide(
    nir_reflectance.add(red_reflectance)
).rename('ndvi')

# Clip NDVI to valid range (-1 to 1) and handle division by zero
# Values outside -1 to 1 are set to -1 (no data)
ndvi = ndvi.clamp(-1, 1)

# Compute NDVI statistics
ndvi_stats = ndvi.reduceRegion(
    reducer=ee.Reducer.percentile([2, 98]).combine(
        ee.Reducer.minMax(), '', True
    ),
    geometry=aoi,
    scale=30,
    maxPixels=1e9
).getInfo()

ndvi_min = ndvi_stats.get('ndvi_min', -1)
ndvi_max = ndvi_stats.get('ndvi_max', 1)
ndvi_p2 = ndvi_stats.get('ndvi_p2', -1)
ndvi_p98 = ndvi_stats.get('ndvi_p98', 1)

print(f"NDVI range: {ndvi_min:.4f} - {ndvi_max:.4f}")
print(f"NDVI 2nd-98th percentile: {ndvi_p2:.4f} - {ndvi_p98:.4f}")
print()

# ===================================================================
# VISUALIZE AND EXPORT NDVI MAP
# ===================================================================
print("=" * 60)
print("CREATING NDVI MAP")
print("=" * 60)

# Create output directory
output_dir = Path(os.path.join('..', 'output'))
output_dir.mkdir(exist_ok=True, parents=True)

# Visualize NDVI with color palette
# Typical NDVI colormap: brown (low) -> yellow -> green (high vegetation)
ndvi_visualized = ndvi.visualize(
    min=ndvi_p2,
    max=ndvi_p98,
    palette=['brown', 'yellow', 'green']
)

# Download function (reuse pattern from HW03)
def download_image(image, filename, aoi, native_crs, scale=30, max_scale=120):
    """Download image with automatic scale adjustment if too large."""
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
            print("This may take a few minutes depending on the image size...")
            
            urllib.request.urlretrieve(download_url, str(filename))
            
            file_size = os.path.getsize(filename) / (1024 * 1024)  # Size in MB
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
                    print(f"  Original error: {e}")
                    return False
            else:
                print(f"  Error: {e}")
                return False
    
    return False

# Download NDVI map
ndvi_filename = output_dir / f"NDVI_{scene_id}.png"
print("Downloading NDVI map...")
ndvi_success = download_image(ndvi_visualized, ndvi_filename, aoi, native_crs, scale=30, max_scale=120)

if not ndvi_success:
    ndvi_filename = None

print()

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("=" * 60)
print("NDVI CALCULATION COMPLETE")
print("=" * 60)
print("NDVI Statistics:")
print(f"  Range: {ndvi_min:.4f} - {ndvi_max:.4f}")
print(f"  2nd-98th percentile: {ndvi_p2:.4f} - {ndvi_p98:.4f}")
print()
print("Downloaded Files:")
if ndvi_filename:
    print(f"  ✓ NDVI map: {ndvi_filename}")
print(f"All files saved in: {output_dir.absolute()}")
print()

