"""Folders used by the analysis scripts. Edit these four paths before running anything; every script reads them here.

AFM      project of the observed-yield analysis: inputs/, outputs/ (DAPSA yields, masks), work/ (exposures)
OUTPUTS  crop masks, department mask and ids, detrended DAPSA yields and series selection
DSSAT    stored DSSAT outputs and analysis folders: pipeline/ (pixels.csv, full-resolution runs), e1/, e2/, x2x3/,
         figures_article/
DATA     ANACIM merged daily grids: arc2/ARC2_merged.nc, chirts/CHIRTS_Tmax_merged.nc, chirts/CHIRTS_Tmin_merged.nc
"""
import os

SCRIPTS = os.path.dirname(os.path.abspath(__file__)) + "/"
AFM = "/home/claude/v6_in/senegal_heat_final/"
OUTPUTS = "/home/claude/outputs/"
DSSAT = "/home/claude/dssat/"
DATA = "/home/claude/data"
