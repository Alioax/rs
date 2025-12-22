"""
Get Landsat 8 and 9 Level 2 images for Iran Wildfires - November 2025

This script:
1. Filters Landsat 8 and 9 Level 2 collections by multiple Path/Row combinations
   (Path 164/Row 35 and Path 165/Row 35)
2. Filters images from November 2025
3. Creates natural color RGB composites for each image
4. Creates false color composites optimized for wildfire detection
5. Downloads all images in the specified period

Output:
- Natural color RGB: True-color visualization (B4, B3, B2)
- Fire detection false color: Highlights fires and burned areas using thermal bands (ST_B10, SR_B7, SR_B4)
"""

import ee
import os
import json
import urllib.request
from pathlib import Path
from datetime import datetime

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

# Configuration
# Path/Row combinations for Iran region
PATH_ROW_COMBINATIONS = [
    (164, 35),
    (165, 35)
]
START_DATE = '2025-11-01'
END_DATE = '2025-11-30'

# Landsat 8 and 9 Level 2 Collections (Collection 2, Tier 1, Level 2)
# Level 2 provides surface reflectance products - better for scientific research
LANDSAT8_COLLECTION = 'LANDSAT/LC08/C02/T1_L2'
LANDSAT9_COLLECTION = 'LANDSAT/LC09/C02/T1_L2'

print("=" * 60)
print("IRAN WILDFIRES - LANDSAT IMAGES NOVEMBER 2025")
print("=" * 60)
print(f"Path/Row combinations: {PATH_ROW_COMBINATIONS}")
print(f"Date Range: {START_DATE} to {END_DATE}")
print(f"Collections: Landsat 8 & 9 Level 2")
print("=" * 60)
print()

# Get the script's directory and navigate to experiment root
# Script is in: experiments/exp_iran_wildfires_nov2025/scripts/
# Experiment root is: experiments/exp_iran_wildfires_nov2025/
script_dir = Path(__file__).parent.absolute()
experiment_root = script_dir.parent  # Go up one level from scripts/ to experiment root

# Create output directory within experiment folder
output_dir = experiment_root / 'output'
output_dir.mkdir(exist_ok=True, parents=True)

# Create data directory for metadata within experiment folder
data_dir = experiment_root / 'data'
data_dir.mkdir(exist_ok=True, parents=True)

print(f"Experiment root: {experiment_root}")
print(f"Output directory: {output_dir}")
print(f"Data directory: {data_dir}")
print()

def download_image(image, filename, aoi, native_crs, scale=30, max_scale=120):
    """
    Download image with automatic scale adjustment if too large.
    Tries progressively larger scales until download succeeds.
    Uses native CRS to maintain proper aspect ratio (square pixels).
    """
    scales_to_try = [scale, scale * 2, scale * 4, max_scale]
    
    for current_scale in scales_to_try:
        try:
            print(f"  Attempting download at {current_scale}m resolution...")
            download_url = image.getDownloadURL({
                'scale': current_scale,
                'crs': native_crs,
                'region': aoi,
                'format': 'PNG'
            })
            
            print(f"  Downloading to: {filename.name}")
            print("  This may take a few minutes depending on the image size...")
            
            # Download the file
            urllib.request.urlretrieve(download_url, str(filename))
            
            file_size = os.path.getsize(filename) / (1024 * 1024)  # Size in MB
            print(f"  ✓ Image saved successfully")
            print(f"    Resolution: {current_scale}m")
            print(f"    File size: {file_size:.2f} MB")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "must be less than or equal to" in error_msg or "Total request size" in error_msg:
                if current_scale < max_scale:
                    print(f"    Image too large at {current_scale}m, trying higher scale...")
                    continue
                else:
                    print(f"    Error: Image still too large even at {max_scale}m resolution.")
                    print(f"    Original error: {e}")
                    return False
            else:
                print(f"    Error: {e}")
                return False
    
    return False

def process_landsat_collection(collection_id, satellite_name, path, row):
    """
    Process a Landsat collection and return images with metadata.
    
    Args:
        collection_id: Landsat collection ID
        satellite_name: Name of satellite (e.g., "Landsat 8")
        path: WRS Path number
        row: WRS Row number
    """
    print(f"\n{'=' * 60}")
    print(f"PROCESSING {satellite_name.upper()} - Path {path}, Row {row}")
    print(f"{'=' * 60}")
    
    # Filter collection
    collection = (ee.ImageCollection(collection_id)
                  .filter(ee.Filter.eq('WRS_PATH', path))
                  .filter(ee.Filter.eq('WRS_ROW', row))
                  .filterDate(START_DATE, END_DATE)
                  .sort('system:time_start'))
    
    # Get count
    count = collection.size().getInfo()
    print(f"Found {count} {satellite_name} images in November 2025")
    
    if count == 0:
        print(f"No {satellite_name} images found for the specified criteria.")
        return []
    
    # Get image list with metadata
    image_list = collection.toList(count)
    images_info = []
    
    for i in range(count):
        image = ee.Image(image_list.get(i))
        
        # Get image properties
        props = image.getInfo()['properties']
        scene_id = props.get('LANDSAT_SCENE_ID', props.get('system:id', 'N/A'))
        date_acquired = props.get('DATE_ACQUIRED', props.get('system:time_start', 'N/A'))
        cloud_cover = props.get('CLOUD_COVER', 'N/A')
        
        # Convert date if it's a timestamp
        if isinstance(date_acquired, (int, float)):
            date_acquired = datetime.fromtimestamp(date_acquired / 1000).strftime('%Y-%m-%d')
        
        images_info.append({
            'image': image,
            'scene_id': scene_id,
            'date_acquired': date_acquired,
            'cloud_cover': cloud_cover,
            'satellite': satellite_name,
            'path': path,
            'row': row
        })
        
        print(f"  [{i+1}/{count}] {scene_id} | {date_acquired} | Cloud: {cloud_cover}%")
    
    return images_info

def create_natural_color_rgb(image, aoi, native_crs, global_min_max=None):
    """
    Create natural color RGB composite from Landsat 8/9 image.
    
    For Landsat 8/9:
    - Band 2: Blue (0.45-0.52 µm)
    - Band 3: Green (0.52-0.60 µm)
    - Band 4: Red (0.63-0.69 µm)
    
    Natural Color RGB: B4 (Red), B3 (Green), B2 (Blue)
    
    Args:
        image: Earth Engine image
        aoi: Area of interest geometry
        native_crs: Native CRS string
        global_min_max: Optional dict with global min/max values for consistent colors
                        Format: {'red': [min, max], 'green': [min, max], 'blue': [min, max]}
    """
    # Select RGB bands for natural color
    rgb_bands = ['SR_B4', 'SR_B3', 'SR_B2']  # Red, Green, Blue (Level 2 uses SR_ prefix)
    
    # Check if SR_ bands exist, if not try without prefix (for some collections)
    try:
        rgb_image = image.select(rgb_bands)
        # Test if bands exist
        _ = rgb_image.bandNames().getInfo()
    except:
        # Try without SR_ prefix
        rgb_bands = ['B4', 'B3', 'B2']
        rgb_image = image.select(rgb_bands)
    
    # Use global statistics if provided, otherwise compute per-image statistics
    if global_min_max:
        red_min, red_max = global_min_max['red']
        green_min, green_max = global_min_max['green']
        blue_min, blue_max = global_min_max['blue']
        print("    Using global statistics for consistent colors...")
    else:
        # Compute percentiles for proper stretching (2nd-98th percentiles)
        print("    Computing statistics for RGB visualization...")
        percentiles = rgb_image.select(rgb_bands).reduceRegion(
            reducer=ee.Reducer.percentile([2, 98]),
            geometry=aoi,
            scale=30,  # 30m resolution for visible bands
            maxPixels=1e9
        ).getInfo()
        
        # Extract min and max values for each band
        red_band = rgb_bands[0]
        green_band = rgb_bands[1]
        blue_band = rgb_bands[2]
        
        red_min = percentiles.get(f'{red_band}_p2', 0)
        red_max = percentiles.get(f'{red_band}_p98', 10000)
        green_min = percentiles.get(f'{green_band}_p2', 0)
        green_max = percentiles.get(f'{green_band}_p98', 10000)
        blue_min = percentiles.get(f'{blue_band}_p2', 0)
        blue_max = percentiles.get(f'{blue_band}_p98', 10000)
    
    print(f"    Band ranges - Red: {red_min:.2f}-{red_max:.2f}, Green: {green_min:.2f}-{green_max:.2f}, Blue: {blue_min:.2f}-{blue_max:.2f}")
    
    # Apply linear stretch to 0-255 range for visualization
    rgb_visualized = rgb_image.visualize(
        bands=rgb_bands,
        min=[red_min, green_min, blue_min],
        max=[red_max, green_max, blue_max],
        gamma=1.2  # Slight gamma correction for better visualization
    )
    
    return rgb_visualized

def create_fire_detection_false_color(image, aoi, native_crs, global_min_max=None):
    """
    Create false color composite optimized for wildfire detection using thermal bands.
    
    Uses Thermal, SWIR2, and Red bands to highlight:
    - Active fires: Bright red/orange/yellow (from thermal band)
    - Burned areas: Dark red/brown (from SWIR2)
    - Vegetation: Dark green
    - Water: Dark blue/black
    
    For Landsat 8/9 Level 2:
    - Thermal (ST_B10): 10.6-11.19 µm - Detects active fires and hot spots (100m resolution, resampled)
    - SWIR2 (SR_B7): 2.11-2.29 µm - Shows burned areas and scars
    - Red (SR_B4): 0.64-0.67 µm - Provides contrast and land/vegetation information
    
    False Color: Thermal (Red), SWIR2 (Green), Red (Blue)
    
    Args:
        image: Earth Engine image
        aoi: Area of interest geometry
        native_crs: Native CRS string
        global_min_max: Optional dict with global min/max values for consistent colors
                        Format: {'thermal': [min, max], 'swir2': [min, max], 'red': [min, max]}
    """
    # Select bands for fire detection false color
    # Thermal, SWIR2, Red combination
    # Thermal bands in Level 2 are ST_B10 (or ST_B11)
    fire_bands = ['ST_B10', 'SR_B7', 'SR_B4']  # Thermal, SWIR2, Red
    
    # Check if bands exist, try alternatives if needed
    try:
        fire_image = image.select(fire_bands)
        # Test if bands exist
        _ = fire_image.bandNames().getInfo()
    except:
        # Try ST_B11 if ST_B10 doesn't exist
        try:
            fire_bands = ['ST_B11', 'SR_B7', 'SR_B4']
            fire_image = image.select(fire_bands)
            _ = fire_image.bandNames().getInfo()
        except:
            # Fallback to non-prefixed bands
            try:
                fire_bands = ['B10', 'B7', 'B4']
                fire_image = image.select(fire_bands)
                _ = fire_image.bandNames().getInfo()
            except:
                # Last fallback: try B11
                fire_bands = ['B11', 'B7', 'B4']
                fire_image = image.select(fire_bands)
    
    # Resample thermal band to 30m to match other bands
    # Thermal bands are 100m native resolution
    thermal_band = fire_bands[0]
    swir2_band = fire_bands[1]
    red_band = fire_bands[2]
    
    # Resample thermal band to 30m
    thermal_resampled = image.select(thermal_band).resample('bilinear').reproject(
        crs=native_crs,
        scale=30
    )
    
    # Select other bands at 30m
    swir2_band_img = image.select(swir2_band)
    red_band_img = image.select(red_band)
    
    # Combine all bands
    fire_image = ee.Image.cat([thermal_resampled, swir2_band_img, red_band_img])
    fire_image = fire_image.rename(fire_bands)
    
    # Use global statistics if provided, otherwise compute per-image statistics
    if global_min_max:
        thermal_min, thermal_max = global_min_max['thermal']
        swir2_min, swir2_max = global_min_max['swir2']
        red_min, red_max = global_min_max['red']
        print("    Using global statistics for consistent colors...")
    else:
        # Compute percentiles for proper stretching (2nd-98th percentiles)
        print("    Computing statistics for fire detection false color...")
        percentiles = fire_image.select(fire_bands).reduceRegion(
            reducer=ee.Reducer.percentile([2, 98]),
            geometry=aoi,
            scale=30,  # 30m resolution
            maxPixels=1e9
        ).getInfo()
        
        # Extract min and max values for each band
        thermal_min = percentiles.get(f'{thermal_band}_p2', 0)
        thermal_max = percentiles.get(f'{thermal_band}_p98', 10000)
        swir2_min = percentiles.get(f'{swir2_band}_p2', 0)
        swir2_max = percentiles.get(f'{swir2_band}_p98', 10000)
        red_min = percentiles.get(f'{red_band}_p2', 0)
        red_max = percentiles.get(f'{red_band}_p98', 10000)
    
    print(f"    Band ranges - Thermal: {thermal_min:.2f}-{thermal_max:.2f}, SWIR2: {swir2_min:.2f}-{swir2_max:.2f}, Red: {red_min:.2f}-{red_max:.2f}")
    
    # Apply linear stretch to 0-255 range for visualization
    fire_visualized = fire_image.visualize(
        bands=fire_bands,
        min=[thermal_min, swir2_min, red_min],
        max=[thermal_max, swir2_max, red_max],
        gamma=1.2  # Slight gamma correction for better visualization
    )
    
    return fire_visualized

# Process all path/row combinations
print("\n" + "=" * 60)
print("SEARCHING FOR LANDSAT IMAGES")
print("=" * 60)

all_images = []

# Process each path/row combination
for path, row in PATH_ROW_COMBINATIONS:
    print(f"\n{'=' * 60}")
    print(f"PROCESSING PATH {path}, ROW {row}")
    print(f"{'=' * 60}")
    
    # Process Landsat 8
    landsat8_images = process_landsat_collection(LANDSAT8_COLLECTION, "Landsat 8", path, row)
    
    # Process Landsat 9
    landsat9_images = process_landsat_collection(LANDSAT9_COLLECTION, "Landsat 9", path, row)
    
    all_images.extend(landsat8_images)
    all_images.extend(landsat9_images)

total_images = len(all_images)

if total_images == 0:
    print("\n" + "=" * 60)
    print("NO IMAGES FOUND")
    print("=" * 60)
    print("No Landsat images found for the specified criteria.")
    print("Please check:")
    print("  - Path/Row combinations")
    print("  - Date range (November 2025)")
    print("  - Collection availability")
    exit(0)

print("\n" + "=" * 60)
print(f"PROCESSING {total_images} IMAGES")
print("=" * 60)

# Calculate global statistics across all images for consistent color representation
print("\n" + "=" * 60)
print("CALCULATING GLOBAL STATISTICS FOR CONSISTENT COLORS")
print("=" * 60)
print("Computing percentiles across all images...")
print("This ensures consistent color representation between different dates.")

# Create a composite from all images for statistics calculation
# Use median composite to reduce impact of clouds and outliers
all_image_objects = [img_info['image'] for img_info in all_images]
image_collection = ee.ImageCollection.fromImages(all_image_objects)

# Get a representative AOI (use first image's geometry)
representative_aoi = all_images[0]['image'].geometry()

# Select RGB bands for natural color
rgb_bands = ['SR_B4', 'SR_B3', 'SR_B2']
try:
    test_image = all_images[0]['image'].select(rgb_bands)
    _ = test_image.bandNames().getInfo()
except:
    rgb_bands = ['B4', 'B3', 'B2']

# Select fire detection bands (Thermal, SWIR2, Red)
fire_bands = ['ST_B10', 'SR_B7', 'SR_B4']  # Thermal, SWIR2, Red
try:
    test_image = all_images[0]['image'].select(fire_bands)
    _ = test_image.bandNames().getInfo()
except:
    try:
        # Try ST_B11 if ST_B10 doesn't exist
        fire_bands = ['ST_B11', 'SR_B7', 'SR_B4']
        test_image = all_images[0]['image'].select(fire_bands)
        _ = test_image.bandNames().getInfo()
    except:
        # Fallback to non-prefixed bands
        try:
            fire_bands = ['B10', 'B7', 'B4']
            test_image = all_images[0]['image'].select(fire_bands)
            _ = test_image.bandNames().getInfo()
        except:
            fire_bands = ['B11', 'B7', 'B4']

# Calculate global statistics for RGB
print("  Computing global RGB statistics...")
rgb_composite = image_collection.select(rgb_bands).median()
rgb_global_stats = rgb_composite.reduceRegion(
    reducer=ee.Reducer.percentile([2, 98]),
    geometry=representative_aoi,
    scale=30,
    maxPixels=1e9
).getInfo()

red_band = rgb_bands[0]
green_band = rgb_bands[1]
blue_band = rgb_bands[2]

rgb_global_min_max = {
    'red': [rgb_global_stats.get(f'{red_band}_p2', 0), rgb_global_stats.get(f'{red_band}_p98', 10000)],
    'green': [rgb_global_stats.get(f'{green_band}_p2', 0), rgb_global_stats.get(f'{green_band}_p98', 10000)],
    'blue': [rgb_global_stats.get(f'{blue_band}_p2', 0), rgb_global_stats.get(f'{blue_band}_p98', 10000)]
}

print(f"  Global RGB ranges:")
print(f"    Red: {rgb_global_min_max['red'][0]:.2f}-{rgb_global_min_max['red'][1]:.2f}")
print(f"    Green: {rgb_global_min_max['green'][0]:.2f}-{rgb_global_min_max['green'][1]:.2f}")
print(f"    Blue: {rgb_global_min_max['blue'][0]:.2f}-{rgb_global_min_max['blue'][1]:.2f}")

# Calculate global statistics for fire detection
print("  Computing global fire detection statistics...")
# Handle thermal band separately (needs resampling to 30m)
thermal_band = fire_bands[0]
swir2_band = fire_bands[1]
red_fire_band = fire_bands[2]

# Get native CRS from first image for resampling
try:
    native_crs_for_stats = all_images[0]['image'].select(0).projection().crs().getInfo()
except:
    native_crs_for_stats = 'EPSG:32639'  # Fallback to UTM Zone 39N

# Resample thermal bands to 30m for all images, then create composite
thermal_collection = image_collection.select(thermal_band).map(
    lambda img: img.resample('bilinear').reproject(
        crs=native_crs_for_stats,
        scale=30
    )
)
thermal_composite = thermal_collection.median()

# Other bands at 30m
other_bands_composite = image_collection.select([swir2_band, red_fire_band]).median()

# Combine for statistics
fire_composite = ee.Image.cat([thermal_composite, other_bands_composite]).rename(fire_bands)

fire_global_stats = fire_composite.reduceRegion(
    reducer=ee.Reducer.percentile([2, 98]),
    geometry=representative_aoi,
    scale=30,
    maxPixels=1e9
).getInfo()

fire_global_min_max = {
    'thermal': [fire_global_stats.get(f'{thermal_band}_p2', 0), fire_global_stats.get(f'{thermal_band}_p98', 10000)],
    'swir2': [fire_global_stats.get(f'{swir2_band}_p2', 0), fire_global_stats.get(f'{swir2_band}_p98', 10000)],
    'red': [fire_global_stats.get(f'{red_fire_band}_p2', 0), fire_global_stats.get(f'{red_fire_band}_p98', 10000)]
}

print(f"  Global fire detection ranges:")
print(f"    Thermal ({thermal_band}): {fire_global_min_max['thermal'][0]:.2f}-{fire_global_min_max['thermal'][1]:.2f}")
print(f"    SWIR2 ({swir2_band}): {fire_global_min_max['swir2'][0]:.2f}-{fire_global_min_max['swir2'][1]:.2f}")
print(f"    Red ({red_fire_band}): {fire_global_min_max['red'][0]:.2f}-{fire_global_min_max['red'][1]:.2f}")

print("\n" + "=" * 60)
print("PROCESSING INDIVIDUAL IMAGES")
print("=" * 60)

# Store metadata for all images
all_metadata = []

# Process each image
for idx, img_info in enumerate(all_images, 1):
    image = img_info['image']
    scene_id = img_info['scene_id']
    date_acquired = img_info['date_acquired']
    cloud_cover = img_info['cloud_cover']
    satellite = img_info['satellite']
    
    print(f"\n[{idx}/{total_images}] Processing: {scene_id}")
    print(f"  Satellite: {satellite}")
    print(f"  Date: {date_acquired}")
    print(f"  Cloud Cover: {cloud_cover}%")
    
    # Get image geometry and native projection
    aoi = image.geometry()
    try:
        native_crs = image.select(0).projection().crs().getInfo()
    except:
        # Fallback to common UTM zone for Iran (Zone 39N)
        native_crs = 'EPSG:32639'
    
    print(f"  Native CRS: {native_crs}")
    print(f"  Path/Row: {img_info['path']}/{img_info['row']}")
    
    # Clean scene ID for filename (remove special characters)
    safe_scene_id = scene_id.replace('/', '_').replace(':', '_')
    path_row_str = f"P{img_info['path']}R{img_info['row']}"
    satellite_str = satellite.replace(' ', '_')
    
    # Create natural color RGB (using global statistics for consistent colors)
    print("  Creating natural color RGB composite...")
    rgb_visualized = create_natural_color_rgb(image, aoi, native_crs, global_min_max=rgb_global_min_max)
    
    # Create natural color filename
    rgb_filename = output_dir / f"RGB_{satellite_str}_{path_row_str}_{safe_scene_id}.png"
    
    # Download natural color image
    print("  Downloading natural color RGB image...")
    rgb_success = download_image(rgb_visualized, rgb_filename, aoi, native_crs, scale=30, max_scale=120)
    
    # Create false color for fire detection (using global statistics for consistent colors)
    print("  Creating fire detection false color composite...")
    fire_visualized = create_fire_detection_false_color(image, aoi, native_crs, global_min_max=fire_global_min_max)
    
    # Create false color filename
    fire_filename = output_dir / f"FireDetection_{satellite_str}_{path_row_str}_{safe_scene_id}.png"
    
    # Download false color image
    print("  Downloading fire detection false color image...")
    fire_success = download_image(fire_visualized, fire_filename, aoi, native_crs, scale=30, max_scale=120)
    
    # Update metadata
    metadata_entry = {
        'scene_id': scene_id,
        'date_acquired': date_acquired,
        'cloud_cover': cloud_cover,
        'satellite': satellite,
        'path': img_info['path'],
        'row': img_info['row'],
        'rgb_filename': rgb_filename.name if rgb_success else None,
        'fire_filename': fire_filename.name if fire_success else None,
        'status': 'success' if (rgb_success or fire_success) else 'failed'
    }
    
    all_metadata.append(metadata_entry)
    
    if rgb_success and fire_success:
        print(f"  ✓ Successfully processed {scene_id} (both composites)")
    elif rgb_success or fire_success:
        print(f"  ⚠ Partially processed {scene_id} (one composite failed)")
    else:
        print(f"  ✗ Failed to download {scene_id}")

# Save metadata to JSON file
metadata_file = data_dir / 'landsat_images_nov2025_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump({
        'path_row_combinations': PATH_ROW_COMBINATIONS,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'total_images': total_images,
        'images': all_metadata
    }, f, indent=2)

print("\n" + "=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)
print(f"Total images processed: {total_images}")
successful = sum(1 for img in all_metadata if img['status'] == 'success')
failed = total_images - successful

# Count composite types
rgb_count = sum(1 for img in all_metadata if img.get('rgb_filename'))
fire_count = sum(1 for img in all_metadata if img.get('fire_filename'))

print(f"  Successful: {successful}")
print(f"  Failed: {failed}")
print(f"  Natural Color RGB composites: {rgb_count}")
print(f"  Fire Detection false color composites: {fire_count}")
print()
print(f"Output directory: {output_dir.absolute()}")
print(f"Metadata saved to: {metadata_file.absolute()}")
print()

if successful > 0:
    print("Successfully downloaded images:")
    for img in all_metadata:
        if img['status'] == 'success':
            print(f"  ✓ {img['satellite']} - P{img['path']}R{img['row']} - {img['date_acquired']}")
            if img.get('rgb_filename'):
                print(f"      RGB: {img['rgb_filename']}")
            if img.get('fire_filename'):
                print(f"      Fire Detection: {img['fire_filename']}")

print()


