# Diffusion Images Directory

## Required Files

Place your diffusion comparison image files here. These files are typically >50MB and are excluded from git tracking.

- **Before Diffusion.tif** - Raw 30m resolution Landsat bands 2-7 (before diffusion processing)
- **Diffusion.tif** - Diffused 15m resolution bands 2-7 (after diffusion processing using panchromatic band)
- **Diffusion.tif.enp** - ENVI header/metadata file (if applicable)

## Data Source

- **Before**: Raw 30m Landsat multispectral bands 2-7
- **After**: Diffused using 15m panchromatic band, resulting in 15m resolution bands 2-7

## Notes

All .tif files in this directory are excluded from git due to their large size (>50MB). Please place your diffusion comparison images here before running the visualization scripts.

The diffusion process enhances spatial resolution from 30m to 15m by using the panchromatic band information.

