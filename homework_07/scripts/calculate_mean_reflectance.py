"""
Calculate Mean Reflectance Across All Bands

This script:
1. Reads the two reflectance GeoTIFF files (6 bands each)
2. Calculates the mean across all 6 bands for each pixel
3. Saves single-band mean reflectance GeoTIFFs
"""

import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Get script directory and paths
script_dir = Path(__file__).parent.absolute()
homework_dir = script_dir.parent
data_dir = homework_dir / 'data'
input_dir = data_dir / 'change_detection_images'
output_dir = data_dir / 'change_detection_images'

print("=" * 60)
print("CALCULATE MEAN REFLECTANCE ACROSS BANDS")
print("=" * 60)
print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print("=" * 60)
print()

# Find reflectance files (only the two reflectance files)
all_tif_files = list(input_dir.glob('*.tif'))
reflectance_files = [f for f in all_tif_files if '_reflectance.tif' in f.name and '_mean_reflectance' not in f.name]

if not reflectance_files:
    print(f"No reflectance files found in: {input_dir}")
    print("Expected files with '_reflectance.tif' suffix")
    exit(1)

print(f"Found {len(reflectance_files)} reflectance file(s) to process")
print()

def extract_scene_id(filename):
    """
    Extract scene ID from filename.
    
    Args:
        filename: Path object or string with filename like 'LC81660342021231LGN00_multispectral_B2-7_reflectance.tif'
    
    Returns:
        str: Scene ID (e.g., 'LC81660342021231LGN00')
    """
    name = Path(filename).stem
    # Remove '_multispectral_B2-7_reflectance' suffix to get scene ID
    scene_id = name.replace('_multispectral_B2-7_reflectance', '')
    return scene_id

# Process each reflectance file
for tif_file in tqdm(reflectance_files, desc="Processing files"):
    print(f"\nProcessing: {tif_file.name}")
    
    # Extract scene ID for output filename
    scene_id = extract_scene_id(tif_file)
    output_file = output_dir / f"{scene_id}_mean_reflectance.tif"
    
    with rasterio.open(tif_file) as src:
        # Get metadata
        profile = src.profile.copy()
        
        # Verify we have 6 bands
        if src.count != 6:
            print(f"  [WARNING] Expected 6 bands, found {src.count} bands")
        
        print(f"  Bands: {src.count}")
        print(f"  Size: {src.width} × {src.height} pixels")
        print(f"  Data type: {src.dtypes[0]}")
        
        # Read all bands at once
        print(f"  Reading all {src.count} bands...")
        all_bands = src.read()  # Shape: [bands, height, width]
        
        # Calculate mean across bands for each pixel
        # axis=0 means average across the first dimension (bands)
        print(f"  Calculating mean across bands...")
        mean_reflectance = np.mean(all_bands, axis=0)  # Shape: [height, width]
        
        # Update profile for single-band output
        profile.update(
            count=1,  # Single band
            dtype=rasterio.float32,
            compress='lzw',
            nodata=None
        )
        
        # Write mean reflectance to output file
        print(f"  Writing mean reflectance to: {output_file.name}")
        with rasterio.open(output_file, 'w', **profile) as dst:
            # Write the mean band (2D array for single band)
            dst.write(mean_reflectance, 1)
            
            # Set band description
            dst.set_band_description(1, 'Mean Reflectance (B2-B7)')
        
        # Verify output and calculate statistics
        with rasterio.open(output_file) as verify:
            mean_data = verify.read(1)
            valid_data = mean_data[~np.isnan(mean_data)]
            
            if len(valid_data) > 0:
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)
                
                print(f"  [OK] Mean reflectance calculation complete")
                print(f"    Output: {output_file.name}")
                print(f"    Statistics:")
                print(f"      Min: {min_val:.6f}")
                print(f"      Max: {max_val:.6f}")
                print(f"      Mean: {mean_val:.6f}")
                print(f"      Std Dev: {std_val:.6f}")
            else:
                print(f"  [WARNING] No valid data in mean reflectance file")

print()
print("=" * 60)
print("MEAN REFLECTANCE CALCULATION SUMMARY")
print("=" * 60)
print(f"Successfully processed {len(reflectance_files)} file(s)")
print(f"Output location: {output_dir}")
print()
print("Mean reflectance files have:")
print("  - Single band with mean of all 6 bands (B2-B7)")
print("  - Float32 data type")
print("  - Same georeferencing as input files")
print("=" * 60)

