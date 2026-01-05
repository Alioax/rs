"""
Select Optimal Temporal Pair for Change Detection

This script:
1. Loads scene metadata from scenes_metadata.json
2. Filters scenes by quality criteria (cloud cover threshold)
3. Finds optimal image pairs from different years
4. Minimizes day-of-year difference for phenological consistency
5. Selects pair with best combination of temporal match and quality
6. Saves selected pair to JSON file
"""

import os
import json
from datetime import datetime
from pathlib import Path

# Clear terminal screen
os.system('cls' if os.name == 'nt' else 'clear')

# Configuration
MAX_CLOUD_COVER = 20  # Maximum cloud cover percentage for quality filtering
MAX_CLOUD_SHADOW = 10  # Maximum cloud shadow percentage (optional filter)
MAX_SNOW_COVER = 5  # Maximum snow cover percentage (optional filter)

# Satellite filtering: List of allowed satellites for pairing
# Options: "Landsat 5 TM", "Landsat 8 OLI", "Landsat 9 OLI-2"
# Set to None or empty list to allow all satellites
# Example: ALLOWED_SATELLITES = ["Landsat 8 OLI", "Landsat 9 OLI-2"]  # Exclude Landsat 5
ALLOWED_SATELLITES = None  # None = allow all satellites
ALLOWED_SATELLITES = ["Landsat 8 OLI", "Landsat 9 OLI-2"]  # Exclude Landsat 5

print("=" * 60)
print("TEMPORAL PAIR SELECTION FOR CHANGE DETECTION")
print("=" * 60)
print(f"Quality Criteria:")
print(f"  - Max Cloud Cover: {MAX_CLOUD_COVER}%")
print(f"  - Max Cloud Shadow: {MAX_CLOUD_SHADOW}%")
print(f"  - Max Snow Cover: {MAX_SNOW_COVER}%")
if ALLOWED_SATELLITES:
    print(f"  - Allowed Satellites: {', '.join(ALLOWED_SATELLITES)}")
else:
    print(f"  - Allowed Satellites: All")
print("=" * 60)
print()

# Get script directory and file paths
script_dir = Path(__file__).parent.absolute()
homework_dir = script_dir.parent
data_dir = homework_dir / 'data'
scenes_file = data_dir / 'scenes_metadata.json'
output_file = data_dir / 'selected_pair.json'

# Create data directory if it doesn't exist
data_dir.mkdir(exist_ok=True, parents=True)

# Load scenes metadata
if not scenes_file.exists():
    print(f"Error: Scenes metadata file not found: {scenes_file}")
    print("Please run search_landsat_scenes.py first to generate scenes_metadata.json")
    exit(1)

print(f"Loading scenes metadata from: {scenes_file}")
try:
    with open(scenes_file, 'r') as f:
        scenes_data = json.load(f)
except Exception as e:
    print(f"Error loading scenes metadata: {e}")
    exit(1)

scenes = scenes_data.get('scenes', [])
search_params = scenes_data.get('search_parameters', {})

print(f"Loaded {len(scenes)} total scenes")
print()

# Step 1: Quality and Satellite Filtering
print("=" * 60)
print("STEP 1: QUALITY AND SATELLITE FILTERING")
print("=" * 60)

quality_filtered = []
for scene in scenes:
    # Check satellite filter
    if ALLOWED_SATELLITES is not None and len(ALLOWED_SATELLITES) > 0:
        satellite = scene.get('satellite')
        if satellite not in ALLOWED_SATELLITES:
            continue
    
    # Check cloud cover
    cloud_cover = scene.get('cloud_cover')
    if cloud_cover is None or cloud_cover > MAX_CLOUD_COVER:
        continue
    
    # Check cloud shadow (optional, skip if None)
    cloud_shadow = scene.get('cloud_shadow')
    if cloud_shadow is not None and cloud_shadow > MAX_CLOUD_SHADOW:
        continue
    
    # Check snow cover (optional, skip if None)
    snow_cover = scene.get('snow_cover')
    if snow_cover is not None and snow_cover > MAX_SNOW_COVER:
        continue
    
    # Must have valid date and day-of-year for pairing
    if scene.get('date_acquired') == 'N/A' or scene.get('day_of_year') is None:
        continue
    
    quality_filtered.append(scene)

print(f"Scenes after quality and satellite filtering: {len(quality_filtered)}")
if ALLOWED_SATELLITES:
    # Count scenes by satellite
    satellite_counts = {}
    for scene in quality_filtered:
        sat = scene.get('satellite', 'Unknown')
        satellite_counts[sat] = satellite_counts.get(sat, 0) + 1
    print("  Breakdown by satellite:")
    for sat, count in sorted(satellite_counts.items()):
        print(f"    - {sat}: {count}")
print()

if len(quality_filtered) < 2:
    print("Error: Not enough quality scenes found for pairing.")
    print(f"Need at least 2 scenes, found {len(quality_filtered)}")
    print("Try adjusting quality thresholds or check scene availability.")
    exit(1)

# Step 2: Pair Selection
print("=" * 60)
print("STEP 2: TEMPORAL PAIR SELECTION")
print("=" * 60)

best_pair = None
best_score = float('inf')
pairs_considered = 0

print("Evaluating temporal pairs...")
for i, scene1 in enumerate(quality_filtered):
    for j, scene2 in enumerate(quality_filtered[i+1:], start=i+1):
        # Both scenes must be from different years
        year1 = scene1.get('year')
        year2 = scene2.get('year')
        
        if year1 is None or year2 is None or year1 == year2:
            continue
        
        # Both must be from July or August (already filtered, but double-check)
        month1 = scene1.get('month')
        month2 = scene2.get('month')
        
        if month1 not in [7, 8] or month2 not in [7, 8]:
            continue
        
        # Calculate day-of-year difference
        doy1 = scene1.get('day_of_year')
        doy2 = scene2.get('day_of_year')
        
        if doy1 is None or doy2 is None:
            continue
        
        doy_diff = abs(doy1 - doy2)
        
        # Calculate combined cloud cover
        cloud1 = scene1.get('cloud_cover', 0)
        cloud2 = scene2.get('cloud_cover', 0)
        combined_cloud = cloud1 + cloud2
        
        # Scoring: prioritize day-of-year match, then cloud cover
        # Lower score is better
        # Weight day-of-year difference more heavily (multiply by 10)
        # Add cloud cover as secondary factor
        score = (doy_diff * 10) + combined_cloud
        
        pairs_considered += 1
        
        if score < best_score:
            best_score = score
            best_pair = {
                'image_1': scene1,
                'image_2': scene2,
                'day_of_year_difference': doy_diff,
                'year_separation': abs(year1 - year2),
                'combined_cloud_cover': combined_cloud,
                'score': score
            }

print(f"Total pairs considered: {pairs_considered}")
print()

if best_pair is None:
    print("Error: No valid temporal pairs found.")
    print("All pairs must be from different years.")
    exit(1)

# Step 3: Display Selected Pair
print("=" * 60)
print("SELECTED OPTIMAL PAIR")
print("=" * 60)

img1 = best_pair['image_1']
img2 = best_pair['image_2']

print(f"\nImage 1 (Earlier Date):")
print(f"  Scene ID: {img1['scene_id']}")
print(f"  Satellite: {img1['satellite']}")
print(f"  Date: {img1['date_acquired']}")
print(f"  Day of Year: {img1['day_of_year']}")
print(f"  Year: {img1['year']}")
print(f"  Cloud Cover: {img1['cloud_cover']:.2f}%" if img1['cloud_cover'] is not None else "  Cloud Cover: N/A")

print(f"\nImage 2 (Later Date):")
print(f"  Scene ID: {img2['scene_id']}")
print(f"  Satellite: {img2['satellite']}")
print(f"  Date: {img2['date_acquired']}")
print(f"  Day of Year: {img2['day_of_year']}")
print(f"  Year: {img2['year']}")
print(f"  Cloud Cover: {img2['cloud_cover']:.2f}%" if img2['cloud_cover'] is not None else "  Cloud Cover: N/A")

print(f"\nPair Characteristics:")
print(f"  Day-of-Year Difference: {best_pair['day_of_year_difference']} days")
print(f"  Year Separation: {best_pair['year_separation']} years")
print(f"  Combined Cloud Cover: {best_pair['combined_cloud_cover']:.2f}%")
print(f"  Selection Score: {best_pair['score']:.2f}")

print()

# Step 4: Save Selected Pair
print("=" * 60)
print("SAVING SELECTED PAIR")
print("=" * 60)

output_data = {
    'search_parameters': search_params,
    'selection_criteria': {
        'max_cloud_cover': MAX_CLOUD_COVER,
        'max_cloud_shadow': MAX_CLOUD_SHADOW,
        'max_snow_cover': MAX_SNOW_COVER,
        'min_year_separation': 1,
        'allowed_satellites': ALLOWED_SATELLITES if ALLOWED_SATELLITES else "All satellites"
    },
    'selection_statistics': {
        'total_scenes': len(scenes),
        'quality_filtered_scenes': len(quality_filtered),
        'pairs_considered': pairs_considered
    },
    'selected_pair': {
        'image_1': {
            'scene_id': img1['scene_id'],
            'satellite': img1['satellite'],
            'date_acquired': img1['date_acquired'],
            'day_of_year': img1['day_of_year'],
            'year': img1['year'],
            'month': img1['month'],
            'day': img1['day'],
            'cloud_cover': img1['cloud_cover'],
            'cloud_cover_land': img1.get('cloud_cover_land'),
            'cloud_shadow': img1.get('cloud_shadow'),
            'snow_cover': img1.get('snow_cover'),
            'image_quality': img1.get('image_quality'),
            'system_id': img1['system_id'],
            'collection_id': img1['collection_id']
        },
        'image_2': {
            'scene_id': img2['scene_id'],
            'satellite': img2['satellite'],
            'date_acquired': img2['date_acquired'],
            'day_of_year': img2['day_of_year'],
            'year': img2['year'],
            'month': img2['month'],
            'day': img2['day'],
            'cloud_cover': img2['cloud_cover'],
            'cloud_cover_land': img2.get('cloud_cover_land'),
            'cloud_shadow': img2.get('cloud_shadow'),
            'snow_cover': img2.get('snow_cover'),
            'image_quality': img2.get('image_quality'),
            'system_id': img2['system_id'],
            'collection_id': img2['collection_id']
        },
        'day_of_year_difference': best_pair['day_of_year_difference'],
        'year_separation': best_pair['year_separation'],
        'combined_cloud_cover': best_pair['combined_cloud_cover'],
        'selection_score': best_pair['score']
    },
    'selection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

try:
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Selected pair saved to: {output_file}")
    print(f"Absolute path: {os.path.abspath(output_file)}")
except Exception as e:
    print(f"Error saving selected pair: {e}")
    print(f"Attempted to save to: {output_file}")
    exit(1)

print()
print("=" * 60)
print("PAIR SELECTION COMPLETE")
print("=" * 60)
print()

