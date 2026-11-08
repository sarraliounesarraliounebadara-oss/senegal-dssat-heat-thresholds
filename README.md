# Statistical Heat Thresholds and Water Stress in Senegal

## Overview

This repository supports a study of statistical heat thresholds, water stress and crop yield responses to warming in Senegal.

The study combines observed departmental yields with DSSAT simulations of millet and groundnut at approximately 4 km resolution over 1986–2016.

It examines whether statistical temperature thresholds reflect physiological heat damage or the combined effects of heat and water stress.

## Study objectives

- Compare heat thresholds estimated from observed and DSSAT-simulated yields.
- Assess threshold sensitivity to changes in selected crop-model heat-response functions.
- Examine the influence of water stress on estimated heat penalties.
- Evaluate statistical estimates of yield responses to a uniform +1 °C warming.
- Assess the effects of spatial aggregation and observational noise.

## Methods

Crop yields were simulated using DSSAT-CSM 4.8.2, with CERES-Millet and CROPGRO-Peanut.

Statistical analyses use growing degree days (GDD), extreme degree days (EDD), panel regression and bootstrap inference.

Heat thresholds were searched between 26 and 40 °C. Statistical tests account for threshold selection.

## Repository contents

This repository is intended to provide Python scripts for post-processing DSSAT outputs, calculating heat exposure, estimating statistical models, performing bootstrap analyses and generating figures and tables.

Scripts will be added after verification.

## Data availability

Observed crop yields were provided by DAPSA. Daily rainfall and temperature data were obtained from ANACIM.

These third-party datasets are not redistributed here. Access requests should be addressed to their respective providers.

DSSAT is available from https://dssat.net.

Reproducing the complete experiments requires the relevant input data, DSSAT configuration and model outputs.

## Authors

Alioune Badara Sarr  
Ibourahima Kébé

## Status

Repository under preparation. Analysis scripts will be added following verification.
