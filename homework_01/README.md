# Homework 1: Landsat 8 Data Download

## Overview

This homework assignment involved downloading a Landsat 8 Collection 2 Level 1 satellite image from the USGS (United States Geological Survey) EarthExplorer platform. This was a data acquisition exercise with no image processing required. The downloaded image covers Path 166, Row 34, which corresponds to the North Iran region. This image is subsequently used and visualized in Homework 2 for ROI selection and color composite analysis.

## Objectives

1. Access USGS EarthExplorer platform
2. Search for Landsat 8 Collection 2 Level 1 data
3. Select and download appropriate satellite imagery
4. Understand the data structure and organization of Landsat Collection 2 products

## Data Source

- **Satellite**: Landsat 8 OLI/TIRS
- **Collection**: Collection 2, Level 1
- **Source**: USGS EarthExplorer
- **Path/Row**: 166/34 (North Iran Region)
- **Format**: GeoTIFF files organized by band

## What is Landsat 8 Collection 2 Level 1?

**Landsat 8** is a joint NASA/USGS satellite mission that provides moderate-resolution (15-100 meter) imagery of Earth's land surface.

**Collection 2** is the second major reprocessing of the Landsat archive, providing:
- Improved radiometric calibration
- Enhanced geometric accuracy
- Consistent data quality across the archive
- Updated metadata and quality assessment bands

**Level 1** products are:
- **Radiometrically corrected**: Digital Numbers (DN) converted to Top-of-Atmosphere (TOA) radiance
- **Geometrically corrected**: Systematic geometric corrections applied
- **Not atmospherically corrected**: Still contains atmospheric effects
- **Ready for further processing**: Can be used for Level 2 surface reflectance generation

## Data Structure

Landsat 8 Collection 2 Level 1 data typically includes:

- **Multispectral Bands**:
  - Band 2: Blue (0.45-0.51 µm)
  - Band 3: Green (0.53-0.59 µm)
  - Band 4: Red (0.64-0.67 µm)
  - Band 5: Near Infrared (0.85-0.88 µm)
  - Band 6: SWIR 1 (1.57-1.65 µm)
  - Band 7: SWIR 2 (2.11-2.29 µm)
- **Panchromatic Band**: Band 8 (0.50-0.68 µm)
- **Thermal Bands**: Band 10 and 11 (10.60-12.51 µm)
- **Quality Assessment (QA) Bands**: Pixel quality, radiometric saturation, etc.

## Download Process

The download process involved:

1. **Registration**: Creating an account on USGS EarthExplorer
2. **Search Criteria**: 
   - Selecting Landsat 8 Collection 2 Level 1 dataset
   - Specifying Path 166, Row 34 (North Iran Region)
   - Setting date range
   - Filtering by cloud cover
3. **Scene Selection**: Choosing appropriate scene(s) based on:
   - Cloud cover percentage
   - Image quality
   - Date of acquisition
   - Geographic coverage
4. **Download**: Retrieving the selected scene(s) from USGS servers

The downloaded image covering Path 166, Row 34 (North Iran Region) is used in Homework 2 for ROI selection and color composite visualization.

## File Organization

Downloaded Landsat 8 Collection 2 Level 1 files are typically organized as:
- Individual GeoTIFF files for each band
- Metadata files (MTL.txt) containing calibration parameters
- Quality assessment files
- Scene identification based on Path/Row and acquisition date

## Notes

- **No Processing Required**: This homework focused solely on data acquisition
- **Data Format**: Files are in GeoTIFF format with embedded georeferencing
- **Metadata**: Important calibration parameters are stored in MTL (Metadata) files
- **Storage**: Downloaded files can be large (several GB per scene)

## Key Takeaways

1. **Data Access**: Understanding how to access satellite data from official sources
2. **Data Selection**: Learning to evaluate and select appropriate imagery based on quality metrics
3. **Data Structure**: Familiarizing with Landsat Collection 2 file organization
4. **Foundation**: This data acquisition step is essential for subsequent processing and analysis
5. **Application**: The downloaded Path 166/Row 34 image is used in Homework 2 for ROI selection and color composite visualization

## References

- [USGS EarthExplorer](https://earthexplorer.usgs.gov/)
- [Landsat 8 Collection 2 Data Format Control Book](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-1-data-format-control-book)
- [Landsat 8 Overview](https://www.usgs.gov/landsat-missions/landsat-8)
