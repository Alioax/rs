"""
Quick verification of converted reflectance files
"""

import rasterio
import numpy as np
from pathlib import Path

# Get script directory and paths
script_dir = Path(__file__).parent.absolute()
homework_dir = script_dir.parent
data_dir = homework_dir / 'data' / 'change_detection_images'

print("=" * 60)
print("VERIFYING CONVERTED REFLECTANCE FILES")
print("=" * 60)
print()

# Find converted files
reflectance_files = list(data_dir.glob('*_reflectance.tif'))

if not reflectance_files:
    print("No converted reflectance files found!")
    exit(1)

for tif_file in reflectance_files:
    print(f"File: {tif_file.name}")
    
    with rasterio.open(tif_file) as src:
        print(f"  Bands: {src.count}")
        print(f"  Data type: {src.dtypes[0]}")
        
        # Check first band
        band_data = src.read(1)
        valid_data = band_data[~np.isnan(band_data)]
        
        if len(valid_data) > 0:
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            mean_val = np.mean(valid_data)
            
            print(f"  Band 1 (SR_B2) statistics:")
            print(f"    Min: {min_val:.6f}")
            print(f"    Max: {max_val:.6f}")
            print(f"    Mean: {mean_val:.6f}")
            
            if 0.0 <= min_val and max_val <= 1.0:
                print(f"  [OK] Values are in correct reflectance range (0-1)")
            else:
                print(f"  [WARNING] Values outside expected range!")
        print()

print("=" * 60)
print("All converted files verified!")
print("=" * 60)

