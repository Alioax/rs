"""
Convert Landsat Level 2 Surface Reflectance GeoTIFFs to 0-1 reflectance values

This script:
1. Reads the downloaded GeoTIFF files
2. Applies the Landsat Collection 2 Level 2 conversion formula:
   Reflectance = (DN × 0.0000275) - 0.2
3. Clips values to 0-1 range
4. Saves converted GeoTIFFs with float32 data type
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

# Landsat Collection 2 Level 2 conversion parameters
SCALE_FACTOR = 0.0000275
OFFSET = -0.2

print("=" * 60)
print("CONVERT LANDSAT LEVEL 2 TO REFLECTANCE (0-1)")
print("=" * 60)
print(f"Conversion formula: Reflectance = (DN × {SCALE_FACTOR}) + ({OFFSET})")
print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print("=" * 60)
print()

# Find all GeoTIFF files (exclude already converted ones)
all_tif_files = list(input_dir.glob('*.tif'))
tif_files = [f for f in all_tif_files if '_reflectance' not in f.stem]

if not tif_files:
    print(f"No GeoTIFF files found to convert in: {input_dir}")
    print(f"(Skipping already converted files)")
    exit(1)

print(f"Found {len(tif_files)} GeoTIFF file(s) to convert")
if len(all_tif_files) > len(tif_files):
    print(f"(Skipping {len(all_tif_files) - len(tif_files)} already converted file(s))")
print()

def convert_to_reflectance(dn_array):
    """
    Convert digital numbers to surface reflectance using Collection 2 Level 2 formula.
    
    Args:
        dn_array: numpy array of digital numbers
    
    Returns:
        numpy array of reflectance values (0-1)
    """
    # Apply conversion formula: Reflectance = (DN × 0.0000275) - 0.2
    reflectance = (dn_array.astype(np.float32) * SCALE_FACTOR) + OFFSET
    
    # Clip to valid reflectance range (0-1)
    reflectance = np.clip(reflectance, 0.0, 1.0)
    
    return reflectance

# Process each file
for tif_file in tqdm(tif_files, desc="Converting files"):
    print(f"\nProcessing: {tif_file.name}")
    
    # Create output filename
    output_file = output_dir / f"{tif_file.stem}_reflectance.tif"
    
    with rasterio.open(tif_file) as src:
        # Get metadata
        profile = src.profile.copy()
        
        # Update data type to float32 for reflectance values (0-1)
        profile.update(
            dtype=rasterio.float32,
            compress='lzw',  # Add compression to save space
            nodata=None  # Remove nodata since we're converting
        )
        
        print(f"  Bands: {src.count}")
        print(f"  Size: {src.width} × {src.height} pixels")
        print(f"  Original data type: {src.dtypes[0]}")
        print(f"  New data type: float32")
        
        # Process each band
        with rasterio.open(output_file, 'w', **profile) as dst:
            for band_idx in tqdm(range(1, src.count + 1), desc="  Converting bands", leave=False):
                # Read band
                band_data = src.read(band_idx)
                
                # Convert to reflectance
                reflectance_data = convert_to_reflectance(band_data)
                
                # Write converted band
                dst.write(reflectance_data, band_idx)
            
            # Copy band descriptions if available
            for i in range(1, src.count + 1):
                if src.descriptions[i-1]:
                    dst.set_band_description(i, src.descriptions[i-1])
        
        # Verify output
        with rasterio.open(output_file) as verify:
            # Check first band statistics
            first_band = verify.read(1)
            valid_data = first_band[~np.isnan(first_band)]
            
            if len(valid_data) > 0:
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                mean_val = np.mean(valid_data)
                
                print(f"  [OK] Conversion complete")
                print(f"    Output: {output_file.name}")
                print(f"    Reflectance range: {min_val:.6f} to {max_val:.6f}")
                print(f"    Mean reflectance: {mean_val:.6f}")
            else:
                print(f"  [WARNING] No valid data in converted file")

print()
print("=" * 60)
print("CONVERSION SUMMARY")
print("=" * 60)
print(f"Successfully converted {len(tif_files)} file(s)")
print(f"Output location: {output_dir}")
print()
print("Converted files have:")
print("  - Pixel values in reflectance format (0-1)")
print("  - Float32 data type")
print("  - Same georeferencing as originals")
print("=" * 60)

