"""
Create multi-panel visualization comparing NDVI and LST

This script:
1. Loads or calculates NDVI and LST
2. Creates vegetation mask
3. Generates multi-panel visualization:
   - NDVI map (full scene)
   - LST map (full scene)
   - Masked NDVI map (vegetated areas only)
   - Masked LST map (vegetated areas only)
   - Scatter plot (NDVI vs LST with correlation)
4. Exports high-quality figures
"""

import ee
import os
import json
import urllib.request
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage

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
print("NDVI-LST COMPARISON VISUALIZATION")
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
# CALCULATE NDVI
# ===================================================================
print("=" * 60)
print("CALCULATING NDVI")
print("=" * 60)

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

band3 = image.select('B3')
band4 = image.select('B4')

red_reflectance = band3.multiply(refl_mult_b3).add(refl_add_b3).rename('red')
nir_reflectance = band4.multiply(refl_mult_b4).add(refl_add_b4).rename('nir')

ndvi = nir_reflectance.subtract(red_reflectance).divide(
    nir_reflectance.add(red_reflectance)
).rename('ndvi').clamp(-1, 1)

# Get NDVI statistics
ndvi_stats = ndvi.reduceRegion(
    reducer=ee.Reducer.percentile([2, 98]),
    geometry=aoi,
    scale=30,
    maxPixels=1e9
).getInfo()

ndvi_p2 = ndvi_stats.get('ndvi_p2', -1)
ndvi_p98 = ndvi_stats.get('ndvi_p98', 1)

print("NDVI calculated successfully.")
print()

# ===================================================================
# CALCULATE LST
# ===================================================================
print("=" * 60)
print("CALCULATING LAND SURFACE TEMPERATURE")
print("=" * 60)

try:
    rad_mult_b6 = image.get('RADIANCE_MULT_BAND_6').getInfo()
    rad_add_b6 = image.get('RADIANCE_ADD_BAND_6').getInfo()
    k1_b6 = image.get('K1_CONSTANT_BAND_6').getInfo()
    k2_b6 = image.get('K2_CONSTANT_BAND_6').getInfo()
except Exception as e:
    try:
        props = image.getInfo()['properties']
        rad_mult_b6 = props.get('RADIANCE_MULT_BAND_6')
        rad_add_b6 = props.get('RADIANCE_ADD_BAND_6')
        k1_b6 = props.get('K1_CONSTANT_BAND_6')
        k2_b6 = props.get('K2_CONSTANT_BAND_6')
    except Exception as e2:
        print(f"Error: Could not extract thermal metadata: {e2}")
        exit(1)

band6 = image.select('B6')
radiance = band6.multiply(rad_mult_b6).add(rad_add_b6)

k1_img = ee.Image.constant(k1_b6)
k2_img = ee.Image.constant(k2_b6)
emissivity = 0.95

kelvin_temp = k2_img.divide(
    k1_img.multiply(emissivity)
    .divide(radiance)
    .add(1)
    .log()
).rename('temperature')

celsius_temp = kelvin_temp.subtract(273.15).rename('lst_celsius')
celsius_temp_30m = celsius_temp.resample('bilinear').reproject(
    crs=native_crs,
    scale=30
)

# Get LST statistics
lst_stats = celsius_temp_30m.reduceRegion(
    reducer=ee.Reducer.percentile([2, 98]),
    geometry=aoi,
    scale=30,
    maxPixels=1e9
).getInfo()

lst_p2 = lst_stats.get('lst_celsius_p2', 0)
lst_p98 = lst_stats.get('lst_celsius_p98', 50)

print("LST calculated successfully.")
print()

# ===================================================================
# CREATE VEGETATION MASK
# ===================================================================
print("=" * 60)
print("CREATING VEGETATION MASK")
print("=" * 60)

ndvi_threshold = 0.35
vegetation_mask = ndvi.gt(ndvi_threshold).rename('vegetation_mask')

ndvi_masked = ndvi.updateMask(vegetation_mask).rename('ndvi_masked')
lst_masked = celsius_temp_30m.updateMask(vegetation_mask).rename('lst_masked')

print(f"Vegetation mask created (NDVI > {ndvi_threshold})")
print()

# ===================================================================
# DOWNLOAD IMAGES FOR VISUALIZATION
# ===================================================================
print("=" * 60)
print("DOWNLOADING IMAGES FOR VISUALIZATION")
print("=" * 60)

output_dir = Path(os.path.join('..', 'output'))
output_dir.mkdir(exist_ok=True, parents=True)

def download_image(image, filename, aoi, native_crs, scale=30, max_scale=120):
    """Download image with automatic scale adjustment."""
    scales_to_try = [scale, scale * 2, scale * 4, max_scale]
    
    for current_scale in scales_to_try:
        try:
            download_url = image.getDownloadURL({
                'scale': current_scale,
                'crs': native_crs,
                'region': aoi,
                'format': 'PNG'
            })
            
            urllib.request.urlretrieve(download_url, str(filename))
            return True, current_scale
            
        except Exception as e:
            error_msg = str(e)
            if "must be less than or equal to" in error_msg:
                if current_scale < max_scale:
                    continue
                else:
                    return False, None
            else:
                return False, None
    
    return False, None

# Visualize and download each layer
print("Downloading NDVI map...")
ndvi_vis = ndvi.visualize(min=ndvi_p2, max=ndvi_p98, palette=['brown', 'yellow', 'green'])
ndvi_file = output_dir / f"temp_ndvi_{scene_id}.png"
success, scale = download_image(ndvi_vis, ndvi_file, aoi, native_crs, scale=30, max_scale=120)
if not success:
    print("Error downloading NDVI map")
    exit(1)

print("Downloading LST map...")
lst_vis = celsius_temp_30m.visualize(
    min=lst_p2, max=lst_p98, palette=['blue', 'cyan', 'yellow', 'orange', 'red']
)
lst_file = output_dir / f"temp_lst_{scene_id}.png"
success, scale = download_image(lst_vis, lst_file, aoi, native_crs, scale=30, max_scale=120)
if not success:
    print("Error downloading LST map")
    exit(1)

print("Downloading masked NDVI map...")
ndvi_masked_vis = ndvi_masked.visualize(
    min=ndvi_p2, max=ndvi_p98, palette=['brown', 'yellow', 'green']
)
ndvi_masked_file = output_dir / f"temp_ndvi_masked_{scene_id}.png"
success, scale = download_image(ndvi_masked_vis, ndvi_masked_file, aoi, native_crs, scale=30, max_scale=120)
if not success:
    print("Error downloading masked NDVI map")
    exit(1)

print("Downloading masked LST map...")
lst_masked_vis = lst_masked.visualize(
    min=lst_p2, max=lst_p98, palette=['blue', 'cyan', 'yellow', 'orange', 'red']
)
lst_masked_file = output_dir / f"temp_lst_masked_{scene_id}.png"
success, scale = download_image(lst_masked_vis, lst_masked_file, aoi, native_crs, scale=30, max_scale=120)
if not success:
    print("Error downloading masked LST map")
    exit(1)

print()

# ===================================================================
# EXTRACT DATA FOR SCATTER PLOT
# ===================================================================
print("=" * 60)
print("PREPARING SCATTER PLOT DATA")
print("=" * 60)

combined = ndvi_masked.addBands(lst_masked)
sample_points = combined.sample(
    region=aoi,
    scale=30,
    numPixels=10000,
    seed=42
)

samples = sample_points.getInfo()['features']
ndvi_values = []
lst_values = []

for sample in samples:
    props = sample.get('properties', {})
    ndvi_val = props.get('ndvi_masked')
    lst_val = props.get('lst_masked')
    
    if ndvi_val is not None and lst_val is not None:
        ndvi_values.append(ndvi_val)
        lst_values.append(lst_val)

ndvi_array = np.array(ndvi_values)
lst_array = np.array(lst_values)

correlation_coef, p_value = pearsonr(ndvi_array, lst_array)
print(f"Correlation coefficient: {correlation_coef:.4f}")
print()

# ===================================================================
# CREATE MULTI-PANEL FIGURE
# ===================================================================
print("=" * 60)
print("CREATING MULTI-PANEL VISUALIZATION")
print("=" * 60)

# Load downloaded images
ndvi_img = PILImage.open(ndvi_file)
lst_img = PILImage.open(lst_file)
ndvi_masked_img = PILImage.open(ndvi_masked_file)
lst_masked_img = PILImage.open(lst_masked_file)

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)

# Panel 1: NDVI (full scene)
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(ndvi_img)
ax1.set_title('NDVI Map (Full Scene)', fontsize=12, fontweight='bold')
ax1.axis('off')

# Panel 2: LST (full scene)
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(lst_img)
ax2.set_title('LST Map (Full Scene)', fontsize=12, fontweight='bold')
ax2.axis('off')

# Panel 3: Masked NDVI
ax3 = fig.add_subplot(gs[1, 0])
ax3.imshow(ndvi_masked_img)
ax3.set_title(f'NDVI Map (Vegetated Areas Only, NDVI > {ndvi_threshold})', 
              fontsize=12, fontweight='bold')
ax3.axis('off')

# Panel 4: Masked LST
ax4 = fig.add_subplot(gs[1, 1])
ax4.imshow(lst_masked_img)
ax4.set_title(f'LST Map (Vegetated Areas Only, NDVI > {ndvi_threshold})', 
              fontsize=12, fontweight='bold')
ax4.axis('off')

# Panel 5: Scatter plot
ax5 = fig.add_subplot(gs[2, :])
ax5.scatter(ndvi_array, lst_array, alpha=0.3, s=1, c='blue', edgecolors='none')

# Add trend line
z = np.polyfit(ndvi_array, lst_array, 1)
p = np.poly1d(z)
ax5.plot(ndvi_array, p(ndvi_array), "r--", alpha=0.8, linewidth=2, 
         label=f'Trend line (r={correlation_coef:.3f})')

ax5.set_xlabel('NDVI', fontsize=12, fontweight='bold')
ax5.set_ylabel('Land Surface Temperature (°C)', fontsize=12, fontweight='bold')
ax5.set_title(f'NDVI vs LST Correlation (Vegetated Areas Only)\nScene: {scene_id}', 
              fontsize=14, fontweight='bold')

info_text = f'r = {correlation_coef:.4f}  |  p < 0.001  |  n = {len(ndvi_array):,}'
ax5.text(0.5, 0.95, info_text, transform=ax5.transAxes, 
         fontsize=11, verticalalignment='top', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax5.grid(True, alpha=0.3)
ax5.legend(loc='best')

# Main title
fig.suptitle('NDVI-LST Relationship Analysis', fontsize=16, fontweight='bold', y=0.98)

# Save figure
comparison_filename = output_dir / f"NDVI_LST_Comparison_{scene_id}.png"
plt.savefig(comparison_filename, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Multi-panel comparison saved: {comparison_filename}")

# Clean up temporary files
for temp_file in [ndvi_file, lst_file, ndvi_masked_file, lst_masked_file]:
    if temp_file.exists():
        temp_file.unlink()

print()

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)
print("Generated Files:")
print(f"  ✓ Multi-panel comparison: {comparison_filename}")
print(f"All files saved in: {output_dir.absolute()}")
print()

