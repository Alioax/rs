"""
Temporary script to check pixel value ranges in downloaded GeoTIFF files
"""

import rasterio
import numpy as np
from pathlib import Path

# Get script directory and paths
script_dir = Path(__file__).parent.absolute()
homework_dir = script_dir.parent
data_dir = homework_dir / 'data' / 'change_detection_images'

print("=" * 60)
print("CHECKING PIXEL VALUE RANGES")
print("=" * 60)
print()

# Find GeoTIFF files
tif_files = list(data_dir.glob('*.tif'))

if not tif_files:
    print(f"No GeoTIFF files found in: {data_dir}")
    exit(1)

# Check first file
tif_file = tif_files[0]
print(f"Checking file: {tif_file.name}")
print()

with rasterio.open(tif_file) as src:
    print(f"Image properties:")
    print(f"  Bands: {src.count}")
    print(f"  Width: {src.width} pixels")
    print(f"  Height: {src.height} pixels")
    print(f"  CRS: {src.crs}")
    print(f"  Data type: {src.dtypes[0]}")
    print()
    
    # Check each band
    for band_idx in range(1, src.count + 1):
        band_data = src.read(band_idx)
        
        # Calculate statistics (excluding NoData values)
        valid_data = band_data[~np.isnan(band_data)]
        
        if len(valid_data) == 0:
            print(f"Band {band_idx}: No valid data")
            continue
            
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        median_val = np.median(valid_data)
        
        print(f"Band {band_idx} (SR_B{band_idx + 1}):")
        print(f"  Min: {min_val:.6f}")
        print(f"  Max: {max_val:.6f}")
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Median: {median_val:.6f}")
        print(f"  Std Dev: {std_val:.6f}")
        
        # Determine value range and test conversion formulas
        if max_val <= 1.0 and min_val >= 0.0:
            print(f"  [OK] Values are in REFLECTANCE format (0-1)")
            print(f"    These are direct surface reflectance values")
        elif max_val <= 10000 and min_val >= 0:
            print(f"  [OK] Values are SCALED (0-10000)")
            print(f"    Divide by 10000 to get reflectance (0-1)")
            # Test conversion
            test_reflectance = mean_val / 10000
            print(f"    Example: Mean DN {mean_val:.0f} -> Reflectance {test_reflectance:.4f}")
        elif max_val <= 65535:
            print(f"  [INFO] Values are in 16-bit range (0-65535)")
            print(f"    Testing conversion formulas:")
            
            # Test Collection 2 Level 2 formula: Reflectance = (DN * 0.0000275) - 0.2
            test_reflectance_c2 = (mean_val * 0.0000275) - 0.2
            test_max_c2 = (max_val * 0.0000275) - 0.2
            print(f"    Formula 1 (C2 L2): Reflectance = (DN * 0.0000275) - 0.2")
            print(f"      Mean: {mean_val:.0f} -> {test_reflectance_c2:.4f}")
            print(f"      Max: {max_val:.0f} -> {test_max_c2:.4f}")
            
            # Test simple division by 10000
            test_reflectance_10k = mean_val / 10000
            test_max_10k = max_val / 10000
            print(f"    Formula 2 (divide by 10000): Reflectance = DN / 10000")
            print(f"      Mean: {mean_val:.0f} -> {test_reflectance_10k:.4f}")
            print(f"      Max: {max_val:.0f} -> {test_max_10k:.4f}")
            
            # Determine which makes more sense
            if test_max_c2 <= 1.0 and test_reflectance_c2 >= 0:
                print(f"    [RECOMMENDED] Formula 1 gives valid range (0-1)")
            elif test_max_10k <= 1.0 and test_reflectance_10k >= 0:
                print(f"    [RECOMMENDED] Formula 2 gives valid range (0-1)")
            else:
                print(f"    [NOTE] Neither formula gives perfect 0-1 range")
                print(f"    Google Earth Engine may export with different scaling")
        else:
            print(f"  [WARNING] Unexpected range")
        print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("Based on the pixel value ranges above, you can determine:")
print("- If values are 0-1: Already in reflectance format (use directly)")
print("- If values are 0-10000: Divide by 10000 to get reflectance")
print("=" * 60)

