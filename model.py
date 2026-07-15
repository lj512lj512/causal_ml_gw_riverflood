# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:27:05 2026

@author: juliu
"""
import os 
import numpy as np
import pandas as pd
import geopandas as gpd
from idFlood import flood_separate 
from pathlib import Path

#%% prepare data
# get precip, discharge
dataFolder = r'\\geodata.geus.dk\DKmodel_users\FloodWarning\Literature_survey\GWH_flood\CAMELS-DK'
stations   = gpd.read_file(os.path.join(dataFolder, 
                                      'Shapefile', 
                                      'CAMELS_DK_304_gauging_stations.shp'))
catchments = gpd.read_file(os.path.join(dataFolder, 
                                      'Shapefile', 
                                      'CAMELS_DK_304_gauging_catchment_boundaries.shp'))
attributes = pd.read_csv(os.path.join(dataFolder, 
                                      'Attibutes', 
                                      'CAMELS_DK_topography.csv'))
attributes = attributes[attributes['gauge_record_pct'] == 100]
stations   = stations[stations['catch_id'].isin(attributes['catch_id'].to_list())]
catchments = catchments[catchments['catch_id'].isin(attributes['catch_id'].to_list())]

for st in stations['catch_id']:
    # get area in km2
    area        = attributes[attributes['catch_id'] == st]['catch_area'].to_list()[0] / (10**6)
    
    # get time series
    filepath = os.path.join(dataFolder, 
                            'Dynamics', 
                            'Gauged_catchments', 
                            f'CAMELS_DK_obs_based_{int(st)}.csv')
    df = pd.read_csv(filepath).set_index(keys = 'time', drop=True)
    df.index = pd.to_datetime(df.index)
    df = df[['precipitation', 'temperature', 'pet', 'DKM_dtp', 'DKM_wcr', 'Qobs']]
    df = df.dropna(axis=0).reindex(
        pd.date_range(
            start="1989-01-02",
            end  ="2019-12-31",
            freq="D"))
    df.to_csv(rf'data/st_{st}.csv')
    break


#%%


df['runoff_mmd'] = df["Qobs"] * 86400 / (area * 1000)
flood_separate(filePath = Path("data"), 
               savePath = Path("data"), 
               catName = str(st), 
               area_cat= area, 
               data= df,
               yarly_check=True, 
               peak_height=None, 
               calculate_baseflow=True,
               qb_threshold=0.5, 
               Qdiff_threshold=0.005, 
               peaks_diff_threshold=2.5, 
               peak_interval_threshold=14)