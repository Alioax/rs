# Experiment: Iran Wildfires - November 2025

## Purpose

This experiment collects and processes Landsat 8 and 9 Level 2 satellite imagery for multiple Path/Row combinations (164/35 and 165/35) in the Iran region during November 2025 to analyze wildfire events.

## Objectives

1. Collect all available Landsat 8 and 9 Level 2 images for November 2025
2. Create natural color RGB composites for visual analysis
3. Create false color composites optimized for wildfire detection
4. Document all available imagery for the time period
5. Prepare data for wildfire detection and analysis

## Data Source

- **Satellites**: Landsat 8 and Landsat 9
- **Collection**: Level 2 (Surface Reflectance) - Collection 2, Tier 1
- **Path/Row Combinations**: 
  - Path 164, Row 35
  - Path 165, Row 35
- **Date Range**: November 1-30, 2025
- **Product Level**: Level 2 (Surface Reflectance) - optimal for scientific research

## Why Level 2?

Level 2 products provide:
- **Surface Reflectance**: Atmospherically corrected data
- **Better for Analysis**: More accurate for scientific research
- **Consistent Quality**: Standardized processing pipeline
- **Fire Detection**: Better suited for detecting burned areas and smoke

## Files

- `scripts/get_landsat_images_nov2025.py` - Main script to download all images
- `data/landsat_images_nov2025_metadata.json` - Metadata for all downloaded images
- `output/` - Natural color RGB and fire detection false color composites

## Usage

```bash
cd scripts
python get_landsat_images_nov2025.py
```

## Output Format

For each image, two composites are created:

1. **Natural Color RGB**:
   - Filename: `RGB_Landsat_8_P164R35_[SCENE_ID].png` or `RGB_Landsat_9_P165R35_[SCENE_ID].png`
   - Bands: Red (B4), Green (B3), Blue (B2)
   - Purpose: True-color visualization similar to what the human eye sees

2. **Fire Detection False Color**:
   - Filename: `FireDetection_Landsat_8_P164R35_[SCENE_ID].png` or `FireDetection_Landsat_9_P165R35_[SCENE_ID].png`
   - Bands: Thermal (ST_B10), SWIR2 (SR_B7), Red (SR_B4)
   - Purpose: Highlights active fires (bright red/orange/yellow from thermal) and burned areas (dark red/brown from SWIR2)

Both composites:
- 30m resolution (resampled if needed)
- Include scene ID, date, path/row, and cloud cover information
- Saved as PNG format

## False Color for Fire Detection

The fire detection false color composite uses thermal bands for enhanced fire detection:
- **Thermal (ST_B10)** in Red channel - Detects active fires and hot spots (10.6-11.19 µm, resampled to 30m)
- **SWIR2 (SR_B7)** in Green channel - Shows burned areas and scars (2.11-2.29 µm)
- **Red (SR_B4)** in Blue channel - Provides contrast and land/vegetation information (0.64-0.67 µm)

**What to look for:**
- 🔥 **Active fires**: Bright red, orange, or yellow spots (from thermal band - shows heat signatures)
- 🟤 **Burned areas**: Dark red or brown regions (from SWIR2 - shows burned vegetation)
- 🌿 **Vegetation**: Dark green areas
- 💧 **Water**: Dark blue or black

**Why thermal bands?**
Thermal infrared bands are essential for fire detection because they:
- Detect heat signatures from active fires that may not be visible in optical bands
- Show hot spots even through smoke
- Provide better temporal coverage (can detect fires at different times of day)
- Complement SWIR bands for comprehensive fire monitoring

## Notes

- Script processes all available images in November 2025 for both path/row combinations
- Images are downloaded at 30m resolution (or higher if too large)
- Metadata is saved to JSON file for reference
- Both Landsat 8 and 9 are processed for maximum temporal coverage

## Next Steps

- Analyze images for wildfire detection
- Compare before/after images
- Create time series analysis
- Detect burned areas using spectral indices

