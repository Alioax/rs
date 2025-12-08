"""
Calculate and plot Land Surface Temperature (LST) in Celsius for Rasht city

This script:
1. Loads the scene ID from selected_scene.json
2. Calculates LST in Celsius using Band 6 (thermal infrared)
3. Clips to Rasht city area
4. Creates and plots a 30m x 30m resolution temperature map
"""

import ee
import os
import json
import urllib.request
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
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

# Load scene information from JSON file
scene_info_file = os.path.join('..', 'data', 'selected_scene.json')
if not os.path.exists(scene_info_file):
    print(f"Error: {scene_info_file} not found.")
    print("Please run load_landsat5_tm_band6.py first to generate the scene selection.")
    exit(1)

with open(scene_info_file, 'r') as f:
    scene_info = json.load(f)

scene_id = scene_info['scene_id']
collection_id = scene_info['collection_id']
date_acquired = scene_info['date_acquired']
cloud_cover = scene_info['cloud_cover']

print("=" * 60)
print("LAND SURFACE TEMPERATURE (LST) - RASHT CITY")
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

# ===================================================================
# DEFINE RASHT CITY BOUNDING BOX
# ===================================================================
print("=" * 60)
print("DEFINING RASHT CITY AREA")
print("=" * 60)

# Rasht city center coordinates (approximately)
# Latitude: 37.2808° N, Longitude: 49.5831° E
rasht_center_lat = 37.2808
rasht_center_lon = 49.5831

# Define bounding box around Rasht (larger area)
# Adjust these values to zoom in/out as needed
lat_buffer = 0.15  # degrees (approximately 16.5km)
lon_buffer = 0.2  # degrees (approximately 22km at this latitude)

rasht_bbox = ee.Geometry.Rectangle([
    rasht_center_lon - lon_buffer,  # West
    rasht_center_lat - lat_buffer,   # South
    rasht_center_lon + lon_buffer,  # East
    rasht_center_lat + lat_buffer   # North
], 'EPSG:4326')

print(f"Rasht city center: {rasht_center_lat}°N, {rasht_center_lon}°E")
print(f"Bounding box: {lat_buffer*2:.3f}° x {lon_buffer*2:.3f}°")
print(f"Approximate area: ~{lat_buffer*2*111:.1f}km x {lon_buffer*2*111:.1f}km")
print()

# Get native projection
native_crs = image.select(0).projection().crs().getInfo()
print(f"Image native CRS: {native_crs}")

# Transform Rasht bounding box to native CRS
rasht_aoi = rasht_bbox.transform(native_crs, 1)
print("Rasht area defined.")
print()

# ===================================================================
# EXTRACT METADATA PARAMETERS
# ===================================================================
print("=" * 60)
print("EXTRACTING METADATA PARAMETERS")
print("=" * 60)

print("Extracting metadata parameters...")
try:
    rad_mult_b6 = image.get('RADIANCE_MULT_BAND_6').getInfo()
    rad_add_b6 = image.get('RADIANCE_ADD_BAND_6').getInfo()
    k1_b6 = image.get('K1_CONSTANT_BAND_6').getInfo()
    k2_b6 = image.get('K2_CONSTANT_BAND_6').getInfo()
    
    print(f"RADIANCE_MULT_BAND_6: {rad_mult_b6}")
    print(f"RADIANCE_ADD_BAND_6: {rad_add_b6}")
    print(f"K1_CONSTANT_BAND_6: {k1_b6}")
    print(f"K2_CONSTANT_BAND_6: {k2_b6}")
    
except Exception as e:
    print(f"Error extracting metadata: {e}")
    print("Trying alternative property names...")
    try:
        props = image.getInfo()['properties']
        rad_mult_b6 = props.get('RADIANCE_MULT_BAND_6')
        rad_add_b6 = props.get('RADIANCE_ADD_BAND_6')
        k1_b6 = props.get('K1_CONSTANT_BAND_6')
        k2_b6 = props.get('K2_CONSTANT_BAND_6')
        
        if not all([rad_mult_b6, rad_add_b6, k1_b6, k2_b6]):
            raise ValueError("Required metadata properties not found")
        
        print(f"RADIANCE_MULT_BAND_6: {rad_mult_b6}")
        print(f"RADIANCE_ADD_BAND_6: {rad_add_b6}")
        print(f"K1_CONSTANT_BAND_6: {k1_b6}")
        print(f"K2_CONSTANT_BAND_6: {k2_b6}")
    except Exception as e2:
        print(f"Error: Could not extract required metadata properties: {e2}")
        print("LST calculation cannot proceed without metadata.")
        exit(1)

print()

# ===================================================================
# CALCULATE LAND SURFACE TEMPERATURE
# ===================================================================
print("=" * 60)
print("CALCULATING LAND SURFACE TEMPERATURE")
print("=" * 60)

# Select Band 6 (thermal infrared)
print("Selecting Band 6 (thermal infrared)...")
band6 = image.select('B6')

# Convert DN to Radiance
print("Converting DN to spectral radiance...")
radiance = band6.multiply(rad_mult_b6).add(rad_add_b6)

# Convert Radiance to Kelvin (with emissivity correction e=0.95)
print("Converting radiance to brightness temperature (Kelvin)...")
print("Using emissivity correction factor: e = 0.95")

k1_img = ee.Image.constant(k1_b6)
k2_img = ee.Image.constant(k2_b6)
emissivity = 0.95

# Calculate: T = K2 / ln((K1 * e / L) + 1)
kelvin_temp = k2_img.divide(
    k1_img.multiply(emissivity)
    .divide(radiance)
    .add(1)
    .log()
)

# Convert Kelvin to Celsius
print("Converting Kelvin to Celsius...")
celsius_temp = kelvin_temp.subtract(273.15).rename('temperature')

# Clip to Rasht city area
print("Clipping to Rasht city area...")
celsius_temp_rasht = celsius_temp.clip(rasht_aoi)

# Compute statistics for Rasht area
print("Computing temperature statistics for Rasht area...")
rasht_stats = celsius_temp_rasht.reduceRegion(
    reducer=ee.Reducer.percentile([2, 98]),
    geometry=rasht_aoi,
    scale=30,  # 30m resolution
    maxPixels=1e9
).getInfo()

rasht_min = rasht_stats.get('temperature_p2', 0)
rasht_max = rasht_stats.get('temperature_p98', 0)
print(f"Temperature range (Celsius): {rasht_min:.2f} - {rasht_max:.2f} °C")
print()

# ===================================================================
# DOWNLOAD RASHT TEMPERATURE DATA
# ===================================================================
print("=" * 60)
print("DOWNLOADING RASHT TEMPERATURE DATA")
print("=" * 60)

# Create output directory (one level up from scripts)
output_dir = Path(os.path.join('..', 'output'))
output_dir.mkdir(exist_ok=True, parents=True)

# Resample to 30m for visualization
print("Resampling to 30m resolution...")
celsius_temp_30m = celsius_temp_rasht.resample('bilinear').reproject(
    crs=native_crs,
    scale=30
)

# Create visualization
celsius_visualized = celsius_temp_30m.visualize(
    min=rasht_min,
    max=rasht_max,
    palette=['blue', 'cyan', 'yellow', 'orange', 'red']
)

# Download function
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

# Download the temperature map
rasht_map_filename = output_dir / f"LST_Rasht_{scene_id}.png"
print("Downloading Rasht temperature map...")
rasht_success = download_image(celsius_visualized, rasht_map_filename, rasht_aoi, native_crs, scale=30, max_scale=120)

if not rasht_success:
    print("Error: Could not download Rasht temperature map.")
    exit(1)

print()

# ===================================================================
# EXTRACT DATA FOR PLOTTING
# ===================================================================
print("=" * 60)
print("EXTRACTING TEMPERATURE DATA FOR PLOTTING")
print("=" * 60)

print("Extracting temperature values at 30m resolution...")
try:
    # Get temperature values as array
    # Use sampleRectangle to get the full array
    temp_array_info = celsius_temp_30m.sampleRectangle(
        region=rasht_aoi,
        defaultValue=-9999
    )
    
    temp_array = np.array(temp_array_info.get('temperature').getInfo())
    
    # Mask invalid values
    temp_array = np.ma.masked_where(temp_array == -9999, temp_array)
    temp_array = np.ma.masked_invalid(temp_array)
    
    print(f"Temperature array shape: {temp_array.shape}")
    print(f"Valid pixels: {np.ma.count(temp_array)}")
    print(f"Temperature range: {np.ma.min(temp_array):.2f} - {np.ma.max(temp_array):.2f} °C")
    print()
    
except Exception as e:
    print(f"Error extracting temperature array: {e}")
    print("Falling back to sampling method...")
    # Fallback: sample points
    sample_points = celsius_temp_30m.sample(
        region=rasht_aoi,
        scale=30,
        numPixels=50000,
        seed=42
    )
    temp_values = np.array(sample_points.aggregate_array('temperature').getInfo())
    temp_values = temp_values[~np.isnan(temp_values)]
    
    print(f"Sampled {len(temp_values)} temperature values")
    print(f"Temperature range: {temp_values.min():.2f} - {temp_values.max():.2f} °C")
    print()
    
    # Create a simple scatter plot instead
    temp_array = None

# ===================================================================
# CREATE RGB NATURAL COLOR MAP
# ===================================================================
print("=" * 60)
print("CREATING RGB NATURAL COLOR MAP")
print("=" * 60)

# For Landsat 5 TM:
# Band 1: Blue (0.45-0.52 µm)
# Band 2: Green (0.52-0.60 µm)
# Band 3: Red (0.63-0.69 µm)
# True Color RGB: B3 (Red), B2 (Green), B1 (Blue)

print("Selecting RGB bands (B3, B2, B1) for natural color...")
rgb_bands = ['B3', 'B2', 'B1']  # Red, Green, Blue
rgb_image = image.select(rgb_bands)

# Clip to Rasht city area
print("Clipping RGB to Rasht city area...")
rgb_rasht = rgb_image.clip(rasht_aoi)

# Compute percentiles for proper stretching
print("Computing statistics for RGB visualization...")
percentiles = rgb_rasht.select(rgb_bands).reduceRegion(
    reducer=ee.Reducer.percentile([2, 98]),
    geometry=rasht_aoi,
    scale=30,  # 30m resolution for visible bands
    maxPixels=1e9
).getInfo()

# Extract min and max values for each band
b3_min = percentiles.get('B3_p2', 0)
b3_max = percentiles.get('B3_p98', 255)
b2_min = percentiles.get('B2_p2', 0)
b2_max = percentiles.get('B2_p98', 255)
b1_min = percentiles.get('B1_p2', 0)
b1_max = percentiles.get('B1_p98', 255)

print(f"Band 3 (Red) range: {b3_min:.2f} - {b3_max:.2f}")
print(f"Band 2 (Green) range: {b2_min:.2f} - {b2_max:.2f}")
print(f"Band 1 (Blue) range: {b1_min:.2f} - {b1_max:.2f}")

# Resample to 30m for visualization (if needed)
rgb_rasht_30m = rgb_rasht.resample('bilinear').reproject(
    crs=native_crs,
    scale=30
)

# Apply linear stretch to 0-255 range for visualization
rgb_visualized = rgb_rasht_30m.visualize(
    bands=rgb_bands,
    min=[b3_min, b2_min, b1_min],
    max=[b3_max, b2_max, b1_max],
    gamma=1.2  # Slight gamma correction for better visualization
)

print("RGB natural color map created.")
print()

# ===================================================================
# EXTRACT RGB DATA FOR PLOTTING
# ===================================================================
print("=" * 60)
print("EXTRACTING RGB DATA FOR PLOTTING")
print("=" * 60)

print("Extracting RGB values at 30m resolution...")
rgb_array = None
try:
    # Get RGB values as array
    rgb_array_info = rgb_rasht_30m.sampleRectangle(
        region=rasht_aoi,
        defaultValue=-9999
    )
    
    # Extract each band
    r_band = np.array(rgb_array_info.get('B3').getInfo())
    g_band = np.array(rgb_array_info.get('B2').getInfo())
    b_band = np.array(rgb_array_info.get('B1').getInfo())
    
    # Mask invalid values
    r_band = np.ma.masked_where(r_band == -9999, r_band)
    g_band = np.ma.masked_where(g_band == -9999, g_band)
    b_band = np.ma.masked_where(b_band == -9999, b_band)
    
    # Normalize to 0-1 range for each band
    r_norm = (r_band - b3_min) / (b3_max - b3_min) if (b3_max - b3_min) > 0 else r_band
    g_norm = (g_band - b2_min) / (b2_max - b2_min) if (b2_max - b2_min) > 0 else g_band
    b_norm = (b_band - b1_min) / (b1_max - b1_min) if (b1_max - b1_min) > 0 else b_band
    
    # Clip to [0, 1] range
    r_norm = np.clip(r_norm, 0, 1)
    g_norm = np.clip(g_norm, 0, 1)
    b_norm = np.clip(b_norm, 0, 1)
    
    # Apply gamma correction
    gamma = 1.2
    r_norm = np.power(r_norm, 1.0/gamma)
    g_norm = np.power(g_norm, 1.0/gamma)
    b_norm = np.power(b_norm, 1.0/gamma)
    
    # Stack into RGB array (height, width, 3)
    rgb_array = np.dstack([r_norm, g_norm, b_norm])
    
    print(f"RGB array shape: {rgb_array.shape}")
    print("RGB natural color data extracted successfully.")
    print()
    
except Exception as e:
    print(f"Error extracting RGB array: {e}")
    print("RGB array will not be available for plotting.")
    rgb_array = None
    print()

# ===================================================================
# PLOT TEMPERATURE MAP AND RGB MAP SIDE BY SIDE
# ===================================================================
print("=" * 60)
print("PLOTTING RASHT TEMPERATURE MAP AND RGB NATURAL COLOR")
print("=" * 60)

# Set matplotlib parameters
mpl.rcParams['figure.dpi'] = 800
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#FF5F05", "#13294B", "#009FD4", "#FCB316", "#006230", "#007E8E", "#5C0E41", "#7D3E13"])

print("Creating side-by-side plot with LST and RGB maps...")

if temp_array is not None and temp_array.size > 0 and rgb_array is not None:
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Ensure both arrays have the same shape (use the smaller dimensions)
    min_height = min(temp_array.shape[0], rgb_array.shape[0])
    min_width = min(temp_array.shape[1], rgb_array.shape[1])
    
    # Crop arrays to same size if needed
    if temp_array.shape != (min_height, min_width):
        temp_array_plot = temp_array[:min_height, :min_width]
    else:
        temp_array_plot = temp_array
    
    if rgb_array.shape[:2] != (min_height, min_width):
        rgb_array_plot = rgb_array[:min_height, :min_width, :]
    else:
        rgb_array_plot = rgb_array
    
    # Plot temperature map on left
    im1 = ax1.imshow(temp_array_plot, 
                     cmap='RdYlBu_r',  # Red-Yellow-Blue reversed (hot to cold)
                     vmin=rasht_min,
                     vmax=rasht_max,
                     interpolation='nearest',
                     aspect='equal',
                     extent=[0, min_width, min_height, 0])  # Set extent for alignment
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add colorbar for temperature
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Temperature (°C)', fontsize=12, fontweight='bold')
    
    # Set labels and title for LST
    ax1.set_xlabel('Pixels (30m resolution)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pixels (30m resolution)', fontsize=12, fontweight='bold')
    ax1.set_title('Land Surface Temperature (LST)', fontsize=14, fontweight='bold', pad=15)
    
    # Add grid
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Add statistics text for LST
    mean_temp = np.ma.mean(temp_array_plot)
    median_temp = np.ma.median(temp_array_plot)
    std_temp = np.ma.std(temp_array_plot)
    
    stats_text = f'Mean: {mean_temp:.2f} °C\nMedian: {median_temp:.2f} °C\nStd Dev: {std_temp:.2f} °C\nRange: {rasht_min:.2f} - {rasht_max:.2f} °C'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot RGB natural color map on right
    ax2.imshow(rgb_array_plot, 
               interpolation='nearest',
               aspect='equal',
               extent=[0, min_width, min_height, 0])  # Same extent for alignment
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Set labels and title for RGB
    ax2.set_xlabel('Pixels (30m resolution)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Pixels (30m resolution)', fontsize=12, fontweight='bold')
    ax2.set_title('RGB Natural Color', fontsize=14, fontweight='bold', pad=15)
    
    # Add grid
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    # Set overall figure title
    fig.suptitle(f'Rasht City - Scene: {scene_id} | Date: {date_acquired}',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot
    plot_filename = output_dir / f"LST_Rasht_Plot_{scene_id}.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    plt.savefig(plot_filename, dpi=800, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Side-by-side plot saved: {plot_filename}")
    print(f"  Mean temperature: {mean_temp:.2f} °C")
    print(f"  Median temperature: {median_temp:.2f} °C")
    print(f"  Standard deviation: {std_temp:.2f} °C")
    
elif temp_array is not None and temp_array.size > 0:
    # Fallback: only LST map if RGB is not available
    print("Warning: RGB array not available. Plotting LST map only.")
    fig, ax = plt.subplots(figsize=(20, 10))
    
    im = ax.imshow(temp_array, 
                   cmap='RdYlBu_r',
                   vmin=rasht_min,
                   vmax=rasht_max,
                   interpolation='nearest',
                   aspect='equal')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°C)', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Pixels (30m resolution)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pixels (30m resolution)', fontsize=12, fontweight='bold')
    ax.set_title(f'Land Surface Temperature (LST) - Rasht City\nScene: {scene_id} | Date: {date_acquired}',
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    mean_temp = np.ma.mean(temp_array)
    median_temp = np.ma.median(temp_array)
    std_temp = np.ma.std(temp_array)
    
    stats_text = f'Mean: {mean_temp:.2f} °C\nMedian: {median_temp:.2f} °C\nStd Dev: {std_temp:.2f} °C\nRange: {rasht_min:.2f} - {rasht_max:.2f} °C'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plot_filename = output_dir / f"LST_Rasht_Plot_{scene_id}.png"
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=800, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Temperature map plot saved: {plot_filename}")
    
else:
    print("Warning: Could not extract temperature array for plotting.")
    print("Using downloaded PNG map instead.")

print()

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("=" * 60)
print("RASHT CITY LST ANALYSIS COMPLETE")
print("=" * 60)
print("Temperature Statistics:")
print(f"  Range: {rasht_min:.2f} - {rasht_max:.2f} °C")
print()
print("Generated Files:")
print(f"  ✓ Rasht temperature map: {rasht_map_filename}")
if 'plot_filename' in locals():
    print(f"  ✓ Rasht temperature plot: {plot_filename}")
print(f"All files saved in: {output_dir.absolute()}")
print()

