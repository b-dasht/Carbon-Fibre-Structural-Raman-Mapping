# Raman Spectra Processing and ID/IG Analysis

This repository contains scripts designed for processing, modelling, and analysing Raman spectra, with a focus on extracting **I<sub>D</sub>/I<sub>G</sub> ratios** that are widely used for assessing carbon-based materials.  

The workflow integrates preprocessing, spectral denoising, and multi-component curve fitting, providing a scalable pipeline for both single spectra and mapped Raman datasets.

The toolbox is customisable and allows automated evaluation of I<sub>D</sub>/I<sub>G</sub> ratios through sequential fitting of D and G bands through a 4 component fitting model.  

The main advantage of this tool lies in its **two stage fitting strategy**, where the D and G bands are initially loosely fitted, before tighter restrictions are autoamtically added along with D2 and G2 bands components. This progressive fitting with parameter restrictions stabilises the fitting process, prevents peak switching, and improves the robustness of I<sub>D</sub>/I<sub>G</sub> quantification.


## Features

- **Preprocessing** (`Preprocess_RamanMap.py`):
  - Loads Raman spectra and converts to structured DataFrame format.
  - Applies **PCA-based noise reduction** to improve signal quality while retaining spectral features.
  - Outputs aligned and denoised spectra with X, Y mapping coordinates.

- **Single Spectrum Modelling** (`Single_Scan_Model.py`):
  - Fits D (~1350 cm⁻¹) and G (~1600 cm⁻¹) peaks using Gaussian models in a two stage process.
  - Calculates:
    - Intensity ratio (I<sub>D</sub>/I<sub>G</sub>).
    - Area ratio (I<sub>D</sub>/I<sub>G</sub>).
    - Goodness-of-fit (R²) with automatic rejection of poor fits.
  - Outputs fitted spectra, parameters, and derived ratios.

- **Automated Map Processing** (`Map_Run.py`):
  - Runs the preprocessing and fitting workflow across all spectra in a map.
  - Saves output files for fitted components, combined fits, and I<sub>D</sub>/I<sub>G</sub> ratio maps.
  - Provides a framework for extending to large Raman imaging datasets.


## Why Use Two Stage Fitting?

- **Prevents unintended shifts** of primary peaks when secondary peaks are present.  
- **Reduces component switching**, ensuring consistent D and G assignments.  
- **Improves repeatability** across mapped datasets.  
- **Supports flexible restrictions** on peak positions, intensities, and widths.  

This makes the workflow more reliable for automated, high-throughput Raman analysis.


## Toolbox Status

This toolbox is functional but no longer being developed. A more powerful and generalised multi-stage mapping spectral analysis pipeline is currently under development to replace this pipeline. Details can be found in my MappingSpectralAnalysis repository.

