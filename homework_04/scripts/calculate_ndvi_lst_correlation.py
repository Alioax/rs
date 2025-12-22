"""
Calculate correlation between NDVI and LST for vegetated areas

This script:
1. Loads scene and recalculates LST (reuse HW03 logic)
2. Calculates NDVI from reflectance
3. Creates vegetation mask (NDVI > 0.35)
4. Applies mask to both NDVI and LST layers
5. Extracts pixel values at 30m resolution
6. Computes Pearson correlation coefficient
7. Generates scatter plot (NDVI vs LST)
8. Exports correlation results and statistics
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
print("NDVI-LST CORRELATION ANALYSIS")
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
# CALCULATE LST (reuse HW03 methodology)
# ===================================================================
print("=" * 60)
print("CALCULATING LAND SURFACE TEMPERATURE")
print("=" * 60)

# Extract thermal metadata
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

# Convert DN to Radiance
band6 = image.select('B6')
radiance = band6.multiply(rad_mult_b6).add(rad_add_b6)

# Convert Radiance to Kelvin (with emissivity correction e=0.95)
k1_img = ee.Image.constant(k1_b6)
k2_img = ee.Image.constant(k2_b6)
emissivity = 0.95

kelvin_temp = k2_img.divide(
    k1_img.multiply(emissivity)
    .divide(radiance)
    .add(1)
    .log()
).rename('temperature')

# Convert Kelvin to Celsius
celsius_temp = kelvin_temp.subtract(273.15).rename('lst_celsius')

# Resample LST to 30m to match NDVI resolution
celsius_temp_30m = celsius_temp.resample('bilinear').reproject(
    crs=native_crs,
    scale=30
)

print("LST calculated successfully (resampled to 30m).")
print()

# ===================================================================
# CREATE VEGETATION MASK
# ===================================================================
print("=" * 60)
print("CREATING VEGETATION MASK")
print("=" * 60)

ndvi_threshold = 0.35
vegetation_mask = ndvi.gt(ndvi_threshold).rename('vegetation_mask')
print(f"Vegetation mask created (NDVI > {ndvi_threshold})")
print()

# ===================================================================
# APPLY MASK TO NDVI AND LST
# ===================================================================
print("=" * 60)
print("APPLYING VEGETATION MASK")
print("=" * 60)

# Mask NDVI and LST (keep only vegetated pixels)
ndvi_masked = ndvi.updateMask(vegetation_mask).rename('ndvi_masked')
lst_masked = celsius_temp_30m.updateMask(vegetation_mask).rename('lst_masked')

print("Mask applied to NDVI and LST layers.")
print()

# ===================================================================
# EXTRACT PIXEL VALUES FOR CORRELATION
# ===================================================================
print("=" * 60)
print("EXTRACTING PIXEL VALUES")
print("=" * 60)

# Create combined image with both NDVI and LST
combined = ndvi_masked.addBands(lst_masked)

# Sample pixels at 30m resolution
# Use a systematic sampling approach to avoid memory issues
print("Sampling pixels for correlation analysis...")
print("This may take a few minutes...")

# Sample the image - use a reasonable sample size
# For large images, we'll sample every Nth pixel to manage memory
sample_scale = 30  # 30m resolution
sample_size = 10000  # Target sample size

# Get image dimensions to determine sampling strategy
image_info = combined.select('ndvi_masked').reduceRegion(
    reducer=ee.Reducer.count(),
    geometry=aoi,
    scale=sample_scale,
    maxPixels=1e9
).getInfo()

total_pixels = image_info.get('ndvi_masked', 0)
print(f"Total masked pixels: {total_pixels:,}")

# Sample pixels
sample_points = combined.sample(
    region=aoi,
    scale=sample_scale,
    numPixels=sample_size,
    seed=42  # For reproducibility
)

# Extract values
print("Extracting sample values...")
samples = sample_points.getInfo()['features']

ndvi_values = []
lst_values = []

for sample in samples:
    props = sample.get('properties', {})
    ndvi_val = props.get('ndvi_masked')
    lst_val = props.get('lst_masked')
    
    # Only include valid (non-null) values
    if ndvi_val is not None and lst_val is not None:
        ndvi_values.append(ndvi_val)
        lst_values.append(lst_val)

ndvi_array = np.array(ndvi_values)
lst_array = np.array(lst_values)

print(f"Extracted {len(ndvi_array):,} valid pixel pairs")
print(f"NDVI range: {ndvi_array.min():.4f} - {ndvi_array.max():.4f}")
print(f"LST range: {lst_array.min():.2f} - {lst_array.max():.2f} °C")
print()

# ===================================================================
# CALCULATE CORRELATION
# ===================================================================
print("=" * 60)
print("CALCULATING CORRELATION")
print("=" * 60)

# Calculate Pearson correlation coefficient
correlation_coef, p_value = pearsonr(ndvi_array, lst_array)

print(f"Pearson correlation coefficient: {correlation_coef:.4f}")
print(f"P-value: {p_value:.2e}")
print(f"Sample size: {len(ndvi_array):,} pixels")
print()

# Interpret correlation strength
if abs(correlation_coef) < 0.3:
    strength = "weak"
elif abs(correlation_coef) < 0.7:
    strength = "moderate"
else:
    strength = "strong"

direction = "negative" if correlation_coef < 0 else "positive"

print(f"Interpretation: {strength} {direction} correlation")
if correlation_coef < 0:
    print("  Higher vegetation density (NDVI) corresponds to lower surface temperature (LST)")
else:
    print("  Higher vegetation density (NDVI) corresponds to higher surface temperature (LST)")
print()

# ===================================================================
# CREATE SCATTER PLOT
# ===================================================================
print("=" * 60)
print("CREATING SCATTER PLOT")
print("=" * 60)

output_dir = Path(os.path.join('..', 'output'))
output_dir.mkdir(exist_ok=True, parents=True)

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 8))

# Use alpha for transparency when many points overlap
ax.scatter(ndvi_array, lst_array, alpha=0.3, s=1, c='blue', edgecolors='none')

# Add trend line
z = np.polyfit(ndvi_array, lst_array, 1)
p = np.poly1d(z)
ax.plot(ndvi_array, p(ndvi_array), "r--", alpha=0.8, linewidth=2, label=f'Trend line (r={correlation_coef:.3f})')

# Labels and title
ax.set_xlabel('NDVI', fontsize=12, fontweight='bold')
ax.set_ylabel('Land Surface Temperature (°C)', fontsize=12, fontweight='bold')
ax.set_title(f'NDVI vs LST Correlation (Vegetated Areas Only)\nScene: {scene_id}', 
             fontsize=14, fontweight='bold')

# Add correlation info as text
info_text = f'r = {correlation_coef:.4f}\np < 0.001\nn = {len(ndvi_array):,}'
ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.grid(True, alpha=0.3)
ax.legend(loc='best')

plt.tight_layout()

# Save scatter plot
scatter_filename = output_dir / f"NDVI_LST_Scatter_{scene_id}.png"
plt.savefig(scatter_filename, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Scatter plot saved: {scatter_filename}")
print()

# ===================================================================
# SAVE CORRELATION RESULTS
# ===================================================================
print("=" * 60)
print("SAVING CORRELATION RESULTS")
print("=" * 60)

results = {
    'scene_id': scene_id,
    'date_acquired': scene_info.get('date_acquired'),
    'ndvi_threshold': ndvi_threshold,
    'correlation_coefficient': float(correlation_coef),
    'p_value': float(p_value),
    'sample_size': int(len(ndvi_array)),
    'ndvi_range': {
        'min': float(ndvi_array.min()),
        'max': float(ndvi_array.max()),
        'mean': float(ndvi_array.mean()),
        'std': float(ndvi_array.std())
    },
    'lst_range': {
        'min': float(lst_array.min()),
        'max': float(lst_array.max()),
        'mean': float(lst_array.mean()),
        'std': float(lst_array.std())
    },
    'interpretation': {
        'strength': strength,
        'direction': direction,
        'description': f"{strength.capitalize()} {direction} correlation between NDVI and LST"
    }
}

results_file = Path(os.path.join('..', 'data', 'correlation_results.json'))
results_file.parent.mkdir(exist_ok=True, parents=True)

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Correlation results saved: {results_file}")
print()

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("=" * 60)
print("CORRELATION ANALYSIS COMPLETE")
print("=" * 60)
print("Results:")
print(f"  Correlation coefficient (r): {correlation_coef:.4f}")
print(f"  P-value: {p_value:.2e}")
print(f"  Sample size: {len(ndvi_array):,} pixels")
print(f"  Interpretation: {strength.capitalize()} {direction} correlation")
print()
print("Downloaded Files:")
print(f"  ✓ Scatter plot: {scatter_filename}")
print(f"  ✓ Results JSON: {results_file}")
print(f"All files saved in: {output_dir.absolute()}")
print()

