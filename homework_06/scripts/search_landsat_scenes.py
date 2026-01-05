"""
Search Landsat Collection 2 Level 1 archive for suitable scenes

This script:
1. Authenticates with Google Earth Engine
2. Searches Landsat 8 and 9 Collection 2 Level 1 collections
3. Filters by WRS Path 162, Row 40
4. Filters for summer acquisitions (July or August)
5. Sorts results by cloud cover (lowest first)
6. Returns all suitable scene IDs with metadata
7. Saves results to JSON file
"""

import ee
import os
import json
from datetime import datetime

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
PATH = 162
ROW = 40
MONTHS = [7, 8]  # July (7) and August (8)

# Landsat 8 and 9 Collection 2 Level 1 (Tier 1) Collections
# Level 1 is the raw/calibrated data (less processed than Level 2)
LANDSAT8_COLLECTION = 'LANDSAT/LC08/C02/T1'
LANDSAT9_COLLECTION = 'LANDSAT/LC09/C02/T1'

print("=" * 60)
print("LANDSAT SCENE SEARCH")
print("=" * 60)
print(f"Path: {PATH}, Row: {ROW}")
print(f"Season: Summer (July and August)")
print(f"Collections: Landsat 8 and Landsat 9 Level 1")
print("=" * 60)
print()

def search_collection(collection_id, satellite_name, path, row, months):
    """
    Search a Landsat collection for suitable scenes.
    
    Args:
        collection_id: Google Earth Engine collection ID
        satellite_name: Name of satellite (e.g., "Landsat 8")
        path: WRS Path number
        row: WRS Row number
        months: List of month numbers (e.g., [7, 8] for July/August)
    
    Returns:
        List of dictionaries containing scene metadata
    """
    print(f"\nSearching {satellite_name} collection...")
    print(f"Collection: {collection_id}")
    
    # Filter collection by Path, Row, and specified months
    # Use calendarRange to filter for July (7) and August (8)
    filtered = (ee.ImageCollection(collection_id)
                .filter(ee.Filter.eq('WRS_PATH', path))
                .filter(ee.Filter.eq('WRS_ROW', row))
                .filter(ee.Filter.calendarRange(months[0], months[-1], 'month'))
                .sort('CLOUD_COVER')  # Sort by cloud cover ascending
                .sort('system:time_start'))  # Then by date
    
    # Get count
    count = filtered.size().getInfo()
    print(f"Found {count} suitable scenes")
    
    if count == 0:
        print(f"No {satellite_name} scenes found for Path {path}, Row {row} in July/August.")
        return []
    
    # Get all scenes as a list
    image_list = filtered.toList(count)
    scenes = []
    
    print(f"Extracting scene metadata...")
    for i in range(count):
        try:
            image = ee.Image(image_list.get(i))
            image_info = image.getInfo()
            props = image_info.get('properties', {})
            
            # Extract scene ID
            scene_id = props.get('LANDSAT_SCENE_ID', 'N/A')
            if scene_id == 'N/A':
                # Try alternative property names
                system_id = props.get('system:id', '')
                if system_id:
                    # Extract scene ID from system ID if available
                    scene_id = system_id.split('/')[-1] if '/' in system_id else system_id
            
            # Extract metadata
            cloud_cover = props.get('CLOUD_COVER', 'N/A')
            date_acquired = props.get('DATE_ACQUIRED', 'N/A')
            if date_acquired == 'N/A':
                # Try alternative date property
                time_start = props.get('system:time_start', 'N/A')
                if time_start != 'N/A':
                    date_acquired = datetime.fromtimestamp(time_start / 1000).strftime('%Y-%m-%d')
            
            system_id = props.get('system:id', 'N/A')
            
            scene_data = {
                'scene_id': scene_id,
                'cloud_cover': cloud_cover if cloud_cover != 'N/A' else None,
                'date_acquired': date_acquired,
                'satellite': satellite_name,
                'system_id': system_id,
                'path': path,
                'row': row,
                'collection_id': collection_id
            }
            
            # Add any additional useful properties
            scene_data['wrs_path'] = props.get('WRS_PATH', path)
            scene_data['wrs_row'] = props.get('WRS_ROW', row)
            scene_data['data_type'] = props.get('DATA_TYPE', 'N/A')
            scene_data['sensor'] = props.get('SENSOR_ID', 'N/A')
            
            scenes.append(scene_data)
            
        except Exception as e:
            print(f"  Warning: Could not extract metadata for image {i+1}: {e}")
            continue
    
    print(f"  Successfully extracted {len(scenes)} scene(s)")
    return scenes

# Search both collections
print("\n" + "=" * 60)
print("SEARCHING COLLECTIONS")
print("=" * 60)

all_scenes = []

# Search Landsat 8
landsat8_scenes = search_collection(
    LANDSAT8_COLLECTION, 
    "Landsat 8", 
    PATH, 
    ROW, 
    MONTHS
)
all_scenes.extend(landsat8_scenes)

# Search Landsat 9
landsat9_scenes = search_collection(
    LANDSAT9_COLLECTION, 
    "Landsat 9", 
    PATH, 
    ROW, 
    MONTHS
)
all_scenes.extend(landsat9_scenes)

# Sort all scenes by cloud cover (ascending), then by date
all_scenes.sort(key=lambda x: (
    x['cloud_cover'] if x['cloud_cover'] is not None and x['cloud_cover'] != 'N/A' else 999,
    x['date_acquired']
))

total_scenes = len(all_scenes)

print("\n" + "=" * 60)
print("SEARCH RESULTS")
print("=" * 60)
print(f"Total suitable scenes found: {total_scenes}")
print()

if total_scenes == 0:
    print("No suitable scenes found for the specified criteria.")
    print("Please check:")
    print("  - Path/Row combination (162/40)")
    print("  - Month filter (July/August)")
    print("  - Collection availability")
    exit(0)

# Display results
print("=" * 60)
print("SUITABLE SCENE IDs")
print("=" * 60)
print(f"{'Scene ID':<25} {'Cloud Cover':<12} {'Date':<12} {'Satellite':<12}")
print("-" * 60)

for scene in all_scenes:
    scene_id = scene['scene_id']
    cloud_cover = scene['cloud_cover']
    date = scene['date_acquired']
    satellite = scene['satellite']
    
    # Format cloud cover
    if cloud_cover is not None and cloud_cover != 'N/A':
        cloud_str = f"{cloud_cover:.2f}%"
    else:
        cloud_str = "N/A"
    
    print(f"{scene_id:<25} {cloud_str:<12} {date:<12} {satellite:<12}")

print("=" * 60)
print()

# Prepare output data
output_data = {
    'search_parameters': {
        'path': PATH,
        'row': ROW,
        'months': MONTHS,
        'month_names': ['July', 'August'],
        'collections': [
            LANDSAT8_COLLECTION,
            LANDSAT9_COLLECTION
        ]
    },
    'total_scenes': total_scenes,
    'search_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'scenes': all_scenes
}

# Save to JSON file
# Get the script's directory and navigate to homework_06 root
script_dir = os.path.dirname(os.path.abspath(__file__))
homework_dir = os.path.dirname(script_dir)  # Go up one level from scripts/ to homework_06/
data_dir = os.path.join(homework_dir, 'data')
output_file = os.path.join(data_dir, 'suitable_scenes.json')

# Create data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Write JSON file with error handling
try:
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {output_file}")
    print(f"Absolute path: {os.path.abspath(output_file)}")
except Exception as e:
    print(f"Error saving JSON file: {e}")
    print(f"Attempted to save to: {output_file}")
    exit(1)

print()

# Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Path/Row: {PATH}/{ROW}")
print(f"Season: July and August")
print(f"Total scenes found: {total_scenes}")
print(f"  - Landsat 8: {len(landsat8_scenes)}")
print(f"  - Landsat 9: {len(landsat9_scenes)}")

if total_scenes > 0:
    # Find best scene (lowest cloud cover)
    best_scene = all_scenes[0]
    print(f"\nBest scene (lowest cloud cover):")
    print(f"  Scene ID: {best_scene['scene_id']}")
    print(f"  Cloud Cover: {best_scene['cloud_cover']:.2f}%" if best_scene['cloud_cover'] is not None else "  Cloud Cover: N/A")
    print(f"  Date: {best_scene['date_acquired']}")
    print(f"  Satellite: {best_scene['satellite']}")

print("=" * 60)
print()

