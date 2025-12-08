"""
Load Landsat 5 TM Level 1 images for Path 166, Row 34, Band 6 (thermal)

This script:
1. Authenticates with Google Earth Engine
2. Filters Landsat 5 TM Level 1 collection by WRS Path 166, Row 34
3. Filters to only summer months (June, July, August)
4. Filters by cloud cover percentage (optional)
5. Identifies the least cloudy summer image
6. Analyzes Band 6 (thermal) value ranges (min/max)
"""

import ee
import os
import json

# Clear terminal screen
os.system('cls' if os.name == 'nt' else 'clear')

# Initialize Google Earth Engine
# Note: Run ee.Authenticate() on first use if not already authenticated
try:
    ee.Initialize(project='surface-hydrology')
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing GEE: {e}")
    print("You may need to run: ee.Authenticate()")
    exit(1)

# Configuration
PATH = 166
ROW = 34
MAX_CLOUD_COVER = 20  # Maximum cloud cover percentage (0-100). Set to None to disable filtering.

# Landsat 5 TM Level 1 (Tier 1) Collection
# Level 1 is the raw/calibrated data (less processed than Level 2)
# Using Collection 2 (C02) as Collection 1 (C01) is deprecated
collection_id = 'LANDSAT/LT05/C02/T1'

print(f"\nSearching for Landsat 5 TM Level 1 images...")
print(f"Path: {PATH}, Row: {ROW}")
print(f"Collection: {collection_id}")
print("Season: Summer (June, July, August)")
if MAX_CLOUD_COVER is not None:
    print(f"Max cloud cover: {MAX_CLOUD_COVER}%")
else:
    print("Cloud cover filter: Disabled")
print()

# Filter collection by Path, Row, and Summer months (June=6, July=7, August=8)
filtered = (ee.ImageCollection(collection_id)
            .filter(ee.Filter.eq('WRS_PATH', PATH))
            .filter(ee.Filter.eq('WRS_ROW', ROW))
            .filter(ee.Filter.calendarRange(6, 8, 'month'))  # Summer: June (6) to August (8)
            .sort('system:time_start'))

# Filter by cloud cover if specified
if MAX_CLOUD_COVER is not None:
    filtered = filtered.filter(ee.Filter.lte('CLOUD_COVER', MAX_CLOUD_COVER))

# Get count
count = filtered.size().getInfo()
print(f"Total summer images found: {count}\n")

if count == 0:
    print("No images found for the specified Path/Row.")
    exit(0)

# Find the least cloudy image
print("Finding the least cloudy summer image...")
least_cloudy = (filtered
                .sort('CLOUD_COVER')
                .first())
least_cloudy_info = least_cloudy.getInfo()
least_cloudy_props = least_cloudy_info.get('properties', {})
least_cloudy_id = least_cloudy_props.get('LANDSAT_SCENE_ID', least_cloudy_props.get('system:id', 'N/A'))
least_cloudy_cloud = least_cloudy_props.get('CLOUD_COVER', 'N/A')
least_cloudy_date = least_cloudy_props.get('DATE_ACQUIRED', least_cloudy_props.get('system:time_start', 'N/A'))

print("=" * 60)
print("LEAST CLOUDY SUMMER IMAGE")
print("=" * 60)
print(f"Scene ID: {least_cloudy_id}")
print(f"Cloud Cover: {least_cloudy_cloud}%")
print(f"Date Acquired: {least_cloudy_date}")
print("=" * 60)
print()

# Save the least cloudy image info to a JSON file for use in other scripts
scene_info = {
    'scene_id': least_cloudy_id,
    'cloud_cover': least_cloudy_cloud,
    'date_acquired': least_cloudy_date,
    'path': PATH,
    'row': ROW,
    'collection_id': collection_id,
    'system_id': least_cloudy_props.get('system:id', 'N/A')
}

# Save to data directory (one level up from scripts)
output_file = os.path.join('..', 'data', 'selected_scene.json')
with open(output_file, 'w') as f:
    json.dump(scene_info, f, indent=2)

print(f"Scene information saved to: {output_file}")
print()

# Analyze Band 6 value ranges
print("Analyzing Band 6 (thermal) value ranges...")
print("Computing min/max across all summer images...\n")

# Select Band 6 from all images
band6_collection = filtered.select('B6')

# Get a sample image to determine the geometry for statistics
sample_img = ee.Image(filtered.first())
aoi = sample_img.geometry()

# Compute min and max across the entire collection
# Create a composite with min and max reducers
band6_min_composite = band6_collection.min()
band6_max_composite = band6_collection.max()

# Get statistics from the composite
stats_min = band6_min_composite.reduceRegion(
    reducer=ee.Reducer.min(),
    geometry=aoi,
    scale=120,  # Band 6 native resolution (120m)
    maxPixels=1e9
).getInfo()

stats_max = band6_max_composite.reduceRegion(
    reducer=ee.Reducer.max(),
    geometry=aoi,
    scale=120,
    maxPixels=1e9
).getInfo()

min_val = stats_min.get('B6', 'N/A')
max_val = stats_max.get('B6', 'N/A')

print("=" * 60)
print("BAND 6 VALUE RANGE ANALYSIS")
print("=" * 60)
print(f"Minimum value: {min_val}")
print(f"Maximum value: {max_val}")
print(f"Range: {min_val} to {max_val}")
print("=" * 60)

# Summary
print(f"\nSummary: {count} summer images found for Path {PATH}, Row {ROW}")
if MAX_CLOUD_COVER is not None:
    print(f"Filtered to images with ≤{MAX_CLOUD_COVER}% cloud cover.")
print("Band 6 (thermal) is available for all images.")

