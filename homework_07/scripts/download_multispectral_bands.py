"""
Export Multispectral Bands for Change Detection to Google Drive

This script:
1. Reads the selected image pair from selected_pair.json
2. Exports multispectral bands (SR_B2 through SR_B7) from each image to Google Drive
3. Saves as georeferenced TIF files at 30m resolution
4. Uses tqdm for export progress tracking
5. Note: Files will be exported to Google Drive in the 'EarthEngine' folder
   You'll need to download them manually from Drive after export completes
"""

import ee
import os
import json
import time
from pathlib import Path
from tqdm import tqdm

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

print("=" * 60)
print("EXPORT MULTISPECTRAL BANDS FOR CHANGE DETECTION")
print("=" * 60)
print("Note: Files will be exported to Google Drive (EarthEngine folder)")
print("      You'll need to download them manually after export completes")
print("=" * 60)
print()

# Get script directory and paths
script_dir = Path(__file__).parent.absolute()
homework_dir = script_dir.parent  # Go up one level from scripts/ to homework_07/
data_dir = homework_dir / 'data'
selected_pair_file = data_dir / 'selected_pair.json'
output_dir = data_dir / 'change_detection_images'

# Create output directory
output_dir.mkdir(exist_ok=True, parents=True)

# Load selected pair
if not selected_pair_file.exists():
    print(f"Error: Selected pair file not found: {selected_pair_file}")
    exit(1)

print(f"Loading selected pair from: {selected_pair_file}")
with open(selected_pair_file, 'r') as f:
    pair_data = json.load(f)

selected_pair = pair_data.get('selected_pair', {})
if not selected_pair:
    print("Error: No selected pair found in JSON file")
    exit(1)

image_1 = selected_pair.get('image_1', {})
image_2 = selected_pair.get('image_2', {})

print(f"Image 1: {image_1.get('scene_id')} ({image_1.get('date_acquired')})")
print(f"Image 2: {image_2.get('scene_id')} ({image_2.get('date_acquired')})")
print()

# Multispectral bands to download (bands 2-7, excluding aerosol band 1)
MULTISPECTRAL_BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
SCALE = 30  # 30 meters resolution

def load_landsat_image(collection_id, scene_id):
    """
    Load a Landsat image from Google Earth Engine by filtering the collection.
    
    Args:
        collection_id: GEE collection ID (e.g., 'LANDSAT/LC08/C02/T1_L2')
        scene_id: Scene ID (e.g., 'LC81660342021231LGN00')
    
    Returns:
        ee.Image: The loaded Landsat image
    """
    try:
        # Filter the collection by scene ID and get the first (and should be only) match
        image = (ee.ImageCollection(collection_id)
                 .filter(ee.Filter.eq('LANDSAT_SCENE_ID', scene_id))
                 .first())
        
        # Verify the image exists by getting basic info
        _ = image.get('system:id').getInfo()
        return image
    except Exception as e:
        print(f"Error loading image {scene_id}: {e}")
        raise

def export_geotiff_to_drive(image, filename, aoi, native_crs, scale=30):
    """
    Export image as GeoTIFF to Google Drive (no size limit).
    
    Args:
        image: ee.Image to export
        filename: Path object for output file (used for naming)
        aoi: Area of interest (geometry)
        native_crs: Native CRS string
        scale: Scale in meters
    
    Returns:
        str: Task ID for monitoring, or None if failed
    """
    try:
        # Create a clean filename for Drive (remove extension)
        drive_filename = filename.stem
        
        print(f"  Creating export task to Google Drive...")
        print(f"  Filename: {drive_filename}.tif")
        print(f"  Resolution: {scale}m")
        print(f"  Location: Google Drive > EarthEngine folder")
        
        # Create export task
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=drive_filename,
            folder='EarthEngine',  # Folder in Google Drive
            fileNamePrefix=drive_filename,
            scale=scale,
            crs=native_crs,
            region=aoi,
            fileFormat='GeoTIFF',
            maxPixels=1e13  # Very large limit to avoid pixel count errors
        )
        
        # Start the task
        task.start()
        task_id = task.id
        print(f"  ✓ Export task started")
        print(f"  Task ID: {task_id}")
        return task_id
        
    except Exception as e:
        print(f"  ✗ Failed to create export task: {e}")
        return None

def wait_for_task_completion(task_id, scene_id):
    """
    Wait for export task to complete with progress tracking.
    
    Args:
        task_id: Task ID string
        scene_id: Scene ID for display
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"  Monitoring export progress for {scene_id}...")
    print(f"  This may take several minutes depending on image size...")
    
    # Monitor task with progress bar
    try:
        with tqdm(desc="    Exporting", unit="check", bar_format='{desc}: {elapsed}') as pbar:
            last_state = None
            
            while True:
                # Refresh task list to get latest state
                task_list = ee.batch.Task.list()
                task = None
                for t in task_list:
                    if t.id == task_id:
                        task = t
                        break
                
                if task is None:
                    print(f"  ✗ Task {task_id} not found")
                    return False
                
                # Get current state - handle both string and EE object cases
                try:
                    if isinstance(task.state, str):
                        state = task.state
                    else:
                        state = task.state.getInfo()
                except:
                    # Fallback: try to get state as string
                    state = str(task.state)
                
                # Update progress bar
                if state != last_state:
                    pbar.set_postfix_str(f"Status: {state}")
                    last_state = state
                
                pbar.update(1)
                
                if state == 'COMPLETED':
                    pbar.set_postfix_str("Status: COMPLETED ✓")
                    print(f"  ✓ Export completed successfully!")
                    print(f"  File is available in Google Drive > EarthEngine folder")
                    return True
                elif state == 'FAILED':
                    try:
                        # Try to get error message
                        if hasattr(task, 'error_message'):
                            if isinstance(task.error_message, str):
                                error_msg = task.error_message
                            else:
                                error_msg = task.error_message.getInfo()
                        else:
                            error_msg = 'Unknown error'
                    except:
                        error_msg = 'Unknown error'
                    print(f"  ✗ Export failed: {error_msg}")
                    return False
                elif state in ['READY', 'RUNNING']:
                    # Task is still running, wait and check again
                    time.sleep(5)
                else:
                    # Unknown state
                    print(f"  ⚠ Unexpected task state: {state}")
                    time.sleep(5)
                    
    except Exception as e:
        print(f"  ✗ Error monitoring task: {e}")
        return False

def process_image(image_info, image_number, total_images):
    """
    Process a single image: load, select bands, and download.
    
    Args:
        image_info: Dictionary with image metadata from selected_pair.json
        image_number: Image number (1 or 2)
        total_images: Total number of images (2)
    """
    scene_id = image_info.get('scene_id')
    collection_id = image_info.get('collection_id')
    date_acquired = image_info.get('date_acquired')
    
    print("=" * 60)
    print(f"PROCESSING IMAGE {image_number}/{total_images}")
    print("=" * 60)
    print(f"Scene ID: {scene_id}")
    print(f"Date: {date_acquired}")
    print(f"Collection: {collection_id}")
    print()
    
    # Load image from GEE
    print("Loading image from Google Earth Engine...")
    try:
        image = load_landsat_image(collection_id, scene_id)
        print("  ✓ Image loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load image: {e}")
        return False
    
    # Select multispectral bands (SR_B2 through SR_B7)
    print(f"Selecting multispectral bands: {', '.join(MULTISPECTRAL_BANDS)}")
    multispectral_image = image.select(MULTISPECTRAL_BANDS)
    print(f"  ✓ Selected {len(MULTISPECTRAL_BANDS)} bands")
    
    # Get image geometry and native CRS
    print("Getting image geometry and projection...")
    aoi = image.geometry()
    try:
        native_crs = image.select(0).projection().crs().getInfo()
    except:
        # Fallback to common UTM zone (adjust if needed)
        native_crs = 'EPSG:32639'  # UTM Zone 39N (common for Iran/Middle East)
        print(f"  Warning: Could not get native CRS, using fallback: {native_crs}")
    else:
        print(f"  Native CRS: {native_crs}")
    
    # Create output filename (used for naming the Drive file)
    safe_scene_id = scene_id.replace('/', '_').replace(':', '_')
    output_filename = output_dir / f"{safe_scene_id}_multispectral_B2-7.tif"
    
    print()
    print(f"Exporting GeoTIFF to Google Drive...")
    task_id = export_geotiff_to_drive(
        multispectral_image,
        output_filename,
        aoi,
        native_crs,
        scale=SCALE
    )
    
    if task_id:
        print()
        success = wait_for_task_completion(task_id, scene_id)
        if success:
            print(f"  ✓ Successfully exported: {output_filename.stem}.tif")
            print(f"  Note: Download the file from Google Drive > EarthEngine folder")
        else:
            print(f"  ✗ Failed to export: {output_filename.stem}.tif")
    else:
        success = False
        print(f"  ✗ Failed to start export task")
    
    print()
    return success

# Process both images
print("=" * 60)
print("STARTING EXPORT PROCESS")
print("=" * 60)
print()

images_to_process = [
    (image_1, 1),
    (image_2, 2)
]

success_count = 0
for image_info, image_num in images_to_process:
    if process_image(image_info, image_num, 2):
        success_count += 1

# Summary
print("=" * 60)
print("EXPORT SUMMARY")
print("=" * 60)
print(f"Successfully exported: {success_count}/2 images")
print(f"Export location: Google Drive > EarthEngine folder")
print("=" * 60)

if success_count == 2:
    print("\n✓ All exports completed successfully!")
    print("\nNext steps:")
    print("1. Go to https://drive.google.com")
    print("2. Navigate to the 'EarthEngine' folder")
    print("3. Download the exported GeoTIFF files")
    print("4. Move them to your desired location")
else:
    print(f"\n⚠ Warning: Only {success_count}/2 images exported successfully.")
    if success_count > 0:
        print("Check Google Drive > EarthEngine folder for completed exports.")
    exit(1)

