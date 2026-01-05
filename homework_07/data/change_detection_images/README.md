# Change Detection Images Directory

## Required Files

Place your change detection image files here. These files are typically >50MB and are excluded from git tracking.

### Raw Multispectral Images

- **LC81660342015231LGN01_multispectral_B2-7.tif** - Raw multispectral bands B2-B7 from 2015 scene
- **LC81660342021231LGN00_multispectral_B2-7.tif** - Raw multispectral bands B2-B7 from 2021 scene

### Reflectance Images

- **LC81660342015231LGN01_multispectral_B2-7_reflectance.tif** - Surface reflectance converted image from 2015 scene
- **LC81660342021231LGN00_multispectral_B2-7_reflectance.tif** - Surface reflectance converted image from 2021 scene

### Processed Images

- **LC81660342015231LGN01_mean_reflectance.tif** - Mean reflectance across all bands for 2015 scene
- **LC81660342021231LGN00_mean_reflectance.tif** - Mean reflectance across all bands for 2021 scene
- **reflectance_difference_2021-2015.tif** - Pixel-wise difference image (2021 - 2015)

## Data Source

- **Location**: Path 166, Row 34 (Northern Iran)
- **Satellites**: Landsat 8 OLI
- **Collection**: LANDSAT/LC08/C02/T1_L2 (Collection 2, Level 2 Surface Reflectance)
- **Selected Pair**:
  - **2015**: LC81660342015231LGN01 (August 19, 2015)
  - **2021**: LC81660342021231LGN00 (August 19, 2021)
- **Bands**: SR_B2 through SR_B7 (multispectral bands, excluding aerosol band)

## Notes

All .tif files in this directory are excluded from git due to their large size (>50MB). 

To obtain these files:
1. Run `scripts/download_multispectral_bands.py` to export from Google Earth Engine to Google Drive
2. Download manually from Google Drive and place in this directory
3. Run `scripts/convert_to_reflectance.py` to convert DN to reflectance
4. Run `scripts/calculate_mean_reflectance.py` to calculate mean reflectance
5. Run `scripts/calculate_reflectance_difference.py` to compute difference image

