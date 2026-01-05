# Homework 6: Landsat Scene Search

## Overview

This homework assignment searches the Landsat Collection 2 Level 1 archive for suitable scenes matching specific criteria using Google Earth Engine.

## Objectives

1. Search Landsat 8 and 9 Collection 2 Level 1 archives
2. Filter by Path 162, Row 40
3. Filter for summer acquisitions (July or August)
4. Identify scenes with minimal to no cloud cover
5. Return suitable scene IDs with metadata (no downloads, no visualization)

## Search Criteria

- **Path/Row**: 162/40
- **Season**: Summer (July or August)
- **Satellites**: Landsat 8 and Landsat 9
- **Collection**: Collection 2, Level 1, Tier 1
- **Cloud Cover**: All scenes sorted by cloud cover (lowest first)
- **Image Quality**: Good quality scenes only

## Files

### Main Script
- `scripts/search_landsat_scenes.py` - Scene search and filtering script

### Data
- `data/suitable_scenes.json` - Generated output containing all suitable scene IDs and metadata

### Outputs
- `output/` - Reserved for future outputs (not used in this assignment)

## Usage

### Option 1: Run Python Script
```bash
cd homework_06/scripts
python search_landsat_scenes.py
```

The script will:
1. Initialize Google Earth Engine
2. Search Landsat 8 and 9 collections
3. Filter by Path 162, Row 40
4. Filter for July/August acquisitions
5. Sort results by cloud cover (lowest first)
6. Display scene IDs in console
7. Save results to `data/suitable_scenes.json`

## Output Format

### Console Output
The script displays:
- Total count of suitable scenes
- List of scene IDs with:
  - Scene ID (LANDSAT_SCENE_ID)
  - Cloud cover percentage
  - Date acquired
  - Satellite mission (Landsat 8 or 9)

### JSON Output
The `suitable_scenes.json` file contains:
- Search parameters (path, row, months, collections)
- Array of suitable scenes with complete metadata:
  - `scene_id`: LANDSAT_SCENE_ID
  - `cloud_cover`: Cloud cover percentage
  - `date_acquired`: Acquisition date
  - `satellite`: Mission name (Landsat 8 or Landsat 9)
  - `system_id`: Google Earth Engine system ID
  - Additional metadata properties

## Example Output Structure

```json
{
  "search_parameters": {
    "path": 162,
    "row": 40,
    "months": [7, 8],
    "collections": [
      "LANDSAT/LC08/C02/T1",
      "LANDSAT/LC09/C02/T1"
    ]
  },
  "total_scenes": 5,
  "scenes": [
    {
      "scene_id": "LC08162402023195LGN00",
      "cloud_cover": 2.5,
      "date_acquired": "2023-07-14",
      "satellite": "Landsat 8",
      "system_id": "LANDSAT/LC08/C02/T1/LC08162402023195LGN00",
      ...
    },
    ...
  ]
}
```

## Data Source

- **Satellites**: Landsat 8 and Landsat 9
- **Collection**: Collection 2, Level 1, Tier 1
  - `LANDSAT/LC08/C02/T1` - Landsat 8
  - `LANDSAT/LC09/C02/T1` - Landsat 9
- **Path/Row**: 162/40
- **Season**: July (month 7) and August (month 8)

## Notes

- Script uses relative paths and should be run from the `scripts/` directory
- Results are saved to `data/suitable_scenes.json`
- No image downloads or visualizations are performed (scene IDs only)
- Scenes are sorted by cloud cover (ascending) for easy identification of best quality images
- All scenes matching the criteria are returned, regardless of cloud cover threshold

## Dependencies

- `earthengine-api` - Google Earth Engine Python API
- `json` - JSON file handling (standard library)

See `requirements.txt` in the repository root for full list.

## Setup

1. Ensure Google Earth Engine is authenticated:
   ```python
   import ee
   ee.Authenticate()
   ee.Initialize(project='your-project-name')
   ```

2. Run the script from the `scripts/` directory:
   ```bash
   cd homework_06/scripts
   python search_landsat_scenes.py
   ```

